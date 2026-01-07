# generator.py (cv2 / numpy version) - 增加超时控制逻辑
import os
import json
import random
import copy
import numpy as np
import cv2
import threading
import time
from typing import Tuple, List, Dict, Any, Optional, Set, FrozenSet
from datetime import datetime
import logging
import traceback
import shutil  # 用于超时后的目录清理

from config import GenConfig
from utils import ensure_dir, save_image
from shapes import Shape
import rules
from sample import populate_prototype
from layout import compose_grid

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class TimeoutException(Exception):
    """自定义超时异常类"""
    pass


class RPMGenerator:
    def __init__(self, config: GenConfig):
        self.cfg = config
        self._init_rng()
        
        # 路径管理
        self.out_dir = self.cfg.out_dir
        self.samples_dir = os.path.join(self.out_dir, "samples")
        self.grids_dir = os.path.join(self.out_dir, "grids")
        ensure_dir(self.samples_dir)
        ensure_dir(self.grids_dir)
        
        # 超时配置 (默认30秒，可通过config设置)
        self.max_generation_time = getattr(self.cfg, 'max_generation_time', 30)  # 单位：秒
        logger.info(f"样本生成超时时间设置为: {self.max_generation_time}秒")

    def _init_rng(self) -> None:
        """初始化随机数生成器"""
        if getattr(self.cfg, 'seed', None) is not None:
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

    # ---------- 辅助方法：颜色转换与画布创建 ----------
    @staticmethod
    def _rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

    @staticmethod
    def _make_canvas(W: int, H: int, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        bgr = RPMGenerator._rgb_to_bgr(bg_color)
        return np.full((H, W, 3), bgr, dtype=np.uint8)

    # ---------- 元素渲染方法 ----------
    def _render_elements_to_canvas(
        self,
        W: int,
        H: int,
        bg_color: Tuple[int, int, int],
        elements: List[Dict[str, Any]],
        use_grid: bool = False,
        grid_size: int = 3
    ) -> np.ndarray:
        """将元素渲染到画布上，支持网格对齐与网格线绘制"""
        canvas = self._make_canvas(W, H, bg_color=bg_color)
        out = canvas.copy()

        # if not elements:
        #     return out

        # 计算网格单元大小
        cell_w, cell_h = W / grid_size, H / grid_size if use_grid else (0, 0)

        for el in elements:
            # 提取元素属性
            kind = el['kind']
            size = int(el['size'])
            fill = bool(el['fill'])
            stroke_width = int(el.get('stroke_width', 1))
            cx, cy = el.get('center', (W//2, H//2))
            color = el.get('color')
            angle = int(el.get('angle', 0))
            flip_mode = el.get('flip_mode')

            # 网格对齐处理
            if use_grid:
                col = min(grid_size-1, max(0, int(cx // cell_w)))
                row = min(grid_size-1, max(0, int(cy // cell_h)))
                cx, cy = int((col + 0.5) * cell_w), int((row + 0.5) * cell_h)

            # 绘制形状
            shape = Shape(kind=kind, size=size, fill=fill, stroke_width=stroke_width)
            try:
                out = shape.draw(out, (cx, cy), angle=angle, color=color, 
                                outline=(0,0,0), flip_mode=flip_mode)
            except TypeError:
                # 兼容不支持flip_mode的Shape.draw接口
                out = shape.draw(out, (cx, cy), angle=angle, color=color, outline=(0,0,0))

        # 绘制网格线
        if use_grid:
            line_color = (0, 0, 0)
            line_thickness = 1
            # 竖线
            for i in range(1, grid_size):
                x = int(round(i * W / grid_size))
                cv2.line(out, (x, 0), (x, H), line_color, line_thickness)
            # 横线
            for j in range(1, grid_size):
                y = int(round(j * H / grid_size))
                cv2.line(out, (0, y), (W, y), line_color, line_thickness)

        return out

    # ---------- 序列化辅助方法 ----------
    @staticmethod
    def _serialize_element(el: Dict[str, Any]) -> Dict[str, Any]:
        """将元素字典转换为JSON可序列化类型"""
        out = {}
        for k, v in el.items():
            if v is None:
                out[k] = None
            elif isinstance(v, (int, float, str, bool)):
                out[k] = v
            elif isinstance(v, (np.integer, np.floating)):
                out[k] = v.item()
            elif isinstance(v, (list, tuple)):
                out[k] = [x.item() if isinstance(x, (np.integer, np.floating)) else x for x in v]
            else:
                out[k] = str(v)  #  fallback
        return out

    @staticmethod
    def _serialize_elements_list(elems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [RPMGenerator._serialize_element(el) for el in elems]

    # ---------- 元素哈希辅助方法（用于去重） ----------
    @staticmethod
    def _element_to_key(elem: Dict[str, Any]) -> Tuple:
        """将元素转换为可哈希的键，递归处理嵌套结构"""
        def _hashable_value(v: Any) -> Any:
            if isinstance(v, dict):
                return tuple(sorted((k, _hashable_value(val)) for k, val in v.items()))
            elif isinstance(v, list) or isinstance(v, tuple):
                return tuple(_hashable_value(item) for item in v)
            elif isinstance(v, set):
                return tuple(sorted(_hashable_value(item) for item in v))
            elif isinstance(v, (np.integer, np.floating)):
                return v.item()
            else:
                try:
                    hash(v)
                    return v
                except TypeError:
                    return str(v)
        
        return tuple(sorted(
            (k, _hashable_value(v)) 
            for k, v in elem.items() 
            if k not in ['timestamp', 'temp_id']
        ))

    @staticmethod
    def _elements_set(elements: List[Dict[str, Any]]) -> FrozenSet:
        """将元素列表转换为可哈希的集合，用于快速去重判断"""
        return frozenset(RPMGenerator._element_to_key(elem) for elem in elements)

    # ---------- 超时控制核心方法 ----------
    def _run_with_timeout(self, func, *args, **kwargs):
        """带超时控制的函数执行器"""
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        # 创建并启动子线程
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        # 等待线程完成或超时
        thread.join(self.max_generation_time)
        
        if thread.is_alive():
            # 线程仍在运行，说明超时
            raise TimeoutException(f"样本生成超时（超过{self.max_generation_time}秒）")
        
        # 如果有异常则抛出
        if exception[0] is not None:
            raise exception[0]
            
        return result[0]

    # ---------- 样本生成核心方法 ----------
    def generate_sample(self, sample_id: int, category_path=None, show_labels=True, show_border=True):
        """生成单个RPM样本，包含超时控制"""
        sample_dir = os.path.join(self.samples_dir, f'sample_{sample_id:06d}')
        grid_path = os.path.join(self.grids_dir, f'grid_{sample_id:06d}.png')
        
        try:
            # 使用超时控制执行实际生成逻辑
            return self._run_with_timeout(
                self._generate_sample_impl,
                sample_id, category_path, show_labels, show_border
            )
        except TimeoutException as e:
            logger.error(f"样本 {sample_id} 生成超时: {str(e)}")
            # 清理超时产生的不完整文件
            self._cleanup_timeout_files(sample_dir, grid_path)
            return None
        except Exception as e:
            logger.error(f"样本 {sample_id} 生成失败: {str(e)}\n{traceback.format_exc()}")
            # 清理失败产生的不完整文件
            self._cleanup_timeout_files(sample_dir, grid_path)
            return None

    def _cleanup_timeout_files(self, sample_dir: str, grid_path: str):
        """清理超时/失败时产生的不完整文件"""
        try:
            if os.path.exists(sample_dir):
                shutil.rmtree(sample_dir)
                logger.info(f"已清理不完整样本目录: {sample_dir}")
            if os.path.exists(grid_path):
                os.remove(grid_path)
                logger.info(f"已清理不完整grid文件: {grid_path}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {str(e)}")

    def _generate_sample_impl(self, sample_id: int, category_path=None, show_labels=True, show_border=True):
        """实际生成逻辑（被超时控制包裹）"""
        start_time = time.time()  # 记录开始时间，用于内部超时检查
        
        # 配置参数获取
        W, H = self.cfg.canvas_size
        num_options = max(1, int(getattr(self.cfg, 'num_options', 4)))
        category_path = category_path or self._sample_category_leaf()
        leaf = category_path[-1]
        handler = rules.RULE_MAP.get(leaf, rules.rule_fallback)
        handler_name = getattr(handler, "__name__", repr(handler))

        # 随机种子设置
        seed_base = (self.cfg.seed or 0) + sample_id
        local_rng = random.Random(seed_base)
        logger.info(f"开始生成样本: id={sample_id}, 规则={leaf}, 处理器={handler_name}, 种子={seed_base}")

        # 确定序列长度
        L = 6 if leaf in ["直接叠加", "去同存异", "去异存同"] else 4

        # 样本目录设置
        sample_dir = os.path.join(self.samples_dir, f'sample_{sample_id:06d}')
        ensure_dir(sample_dir)

        # 是否使用网格
        # use_grid = leaf in ["直接叠加", "去同存异", "去异存同", "平移", "旋转", "翻转(镜像)", "组合"] and random.choice([False, True])
        use_grid = random.choice([False, True])

        # 检查是否超时（初始阶段）
        self._check_timeout(start_time, sample_id)

        # 1. 生成初始状态
        init_elements, init_img = self._create_initial_state(seed_base, W, H, use_grid, leaf, sample_dir)
        self._check_timeout(start_time, sample_id)

        # 2. 生成后续状态序列
        states_internal, history_elements = self._generate_subsequent_states(
            L, W, H, use_grid, seed_base, init_elements, init_img, 
            sample_dir, handler, handler_name, leaf, start_time
        )
        self._check_timeout(start_time, sample_id)

        # 3. 生成正确选项和干扰项
        candidates_internal = self._generate_candidates(
            num_options, W, H, use_grid, seed_base, history_elements, 
            states_internal, sample_dir, handler, handler_name, leaf, 
            local_rng, start_time
        )
        self._check_timeout(start_time, sample_id)

        # 4. 组合网格图像并保存到单独目录
        grid_path, grid_meta = self._compose_and_save_grid(
            sample_id, W, H, states_internal, candidates_internal, 
            sample_dir, num_options, show_labels, show_border
        )
        correct_index = grid_meta['correct_index']

        # 5. 生成并保存元数据
        meta = self._generate_metadata(
            sample_id, category_path, sample_dir, grid_path, 
            states_internal, candidates_internal, correct_index, 
            leaf, seed_base, grid_meta
        )

        logger.info(
            f"样本生成完成: id={sample_id}, "
            f"耗时={time.time()-start_time:.2f}秒, "
            f"保存路径={sample_dir}, "
            f"grid路径={grid_path}"
        )
        return meta

    def _check_timeout(self, start_time: float, sample_id: int):
        """检查是否超时，在关键步骤之间调用"""
        elapsed = time.time() - start_time
        if elapsed > self.max_generation_time:
            raise TimeoutException(
                f"样本 {sample_id} 生成超时（已耗时{elapsed:.2f}秒，超过{self.max_generation_time}秒）"
            )

    def _create_initial_state(self, seed_base: int, W: int, H: int, use_grid: bool, 
                             leaf: str, sample_dir: str) -> Tuple[List[Dict], np.ndarray]:
        """创建初始状态元素和图像"""
        if leaf in ["单一遍历", "位置遍历"]:
            sample_num = 2
        elif leaf in ["平移", "旋转", "翻转(镜像)"]:
            sample_num = 1
        else:
            sample_num = random.randint(1, 3)
        init_state = populate_prototype(
            W, H, bg_color=self.cfg.bg_color, use_grid=use_grid,
            seed=seed_base * 100 + 0, sample_num=sample_num
        )
        init_elements = init_state['elements']
        init_img = self._render_elements_to_canvas(W, H, self.cfg.bg_color, init_elements, use_grid)
        
        # 保存初始状态图像
        state0_path = os.path.join(sample_dir, f'state_0.png')
        save_image(init_img, state0_path)
        
        return init_elements, init_img

    def _generate_subsequent_states(self, L: int, W: int, H: int, use_grid: bool, 
                                   seed_base: int, init_elements: List[Dict], 
                                   init_img: np.ndarray, sample_dir: str, 
                                   handler, handler_name: str, leaf: str,
                                   start_time: float) -> Tuple[List[Dict], List[List[Dict]]]:
        """生成后续状态序列，增加内部超时检查"""
        states_internal = [{
            'state_img': init_img,
            'state_path': os.path.join(sample_dir, f'state_0.png'),
            'elements': copy.deepcopy(init_elements),
            'canvas_size': (W, H),
            'rule_info': None,
            'timestamp': datetime.utcnow().isoformat()
        }]
        history_elements = [copy.deepcopy(init_elements)]
        handler_exc = None
        rule_info = None

        for i in range(1, L):
            # 每步都检查超时
            self._check_timeout(start_time, sample_id=seed_base - (self.cfg.seed or 0))
            
            logger.debug(f"生成第 {i}/{L-1} 步状态")
            next_elements = None
            try:
                res = handler(history_elements, rule_info=rule_info, use_grid=use_grid, config=self.cfg)
                next_elements, rule_info = res
                handler_exc = None
                logger.debug(f"第 {i} 步处理器执行成功: {rule_info}")
            except Exception as e:
                handler_exc = e
                tb = traceback.format_exc()
                logger.error(f"第 {i} 步处理器执行失败: {str(e)}\n{tb}")
                rule_info = {
                    'error': str(e),
                    'traceback': tb,
                    'fallback': True,
                    'handler': handler_name
                }
                next_elements = copy.deepcopy(history_elements[-1])  #  fallback

            # 渲染并保存状态
            next_img = self._render_elements_to_canvas(W, H, self.cfg.bg_color, next_elements, use_grid)
            state_path = os.path.join(sample_dir, f'state_{i}.png')
            save_image(next_img, state_path)

            # 更新状态列表
            history_elements.append(copy.deepcopy(next_elements))
            states_internal.append({
                'state_img': next_img,
                'state_path': state_path,
                'elements': copy.deepcopy(next_elements),
                'canvas_size': (W, H),
                'rule_info': rule_info,
                'timestamp': datetime.utcnow().isoformat()
            })

        return states_internal, history_elements

    def _generate_candidates(self, num_options: int, W: int, H: int, use_grid: bool, 
                            seed_base: int, history_elements: List[List[Dict]], 
                            states_internal: List[Dict], sample_dir: str, 
                            handler, handler_name: str, leaf: str, 
                            local_rng: random.Random, start_time: float) -> List[Dict]:
        """生成正确选项和干扰项，增加内部超时检查"""
        # 正确选项
        true_next_elements = states_internal[-1]['elements']
        true_next_img = states_internal[-1]['state_img']
        true_next_path = os.path.join(sample_dir, 'proto_true_next.png')
        save_image(true_next_img, true_next_path)
        
        candidates_internal = [{
            'img': true_next_img,
            'path': true_next_path,
            'is_correct': True,
            'elements': true_next_elements,
            'rule_info': states_internal[-1].get('rule_info')
        }]

        # 干扰项生成
        history_for_option = copy.deepcopy(history_elements[:-1])  # 移除最后一帧
        option_states = [true_next_elements]
        max_retries = getattr(self.cfg, 'max_distractor_retries', 20)
        sample_id = seed_base - (self.cfg.seed or 0)  # 还原样本ID

        for j in range(1, num_options):
            if len(candidates_internal) >= num_options:
                break

            # 检查超时
            self._check_timeout(start_time, sample_id)
            
            distractor_seed = seed_base * 100 + 2000 + j
            dp_elements = None
            dp_rule_info = None

            # 生成干扰项元素
            retry_count = 0
            while retry_count < max_retries:
                # 每次重试都检查超时
                self._check_timeout(start_time, sample_id)
                
                try:
                    if leaf in ["直接叠加", "去同存异", "去异存同"]:
                        prev1 = history_for_option[-1] if history_for_option else []
                        prev2 = history_for_option[-2] if len(history_for_option)>=2 else []
                        dp_elements = (random.sample(prev1, random.randint(0, len(prev1))) +
                                      random.sample(prev2, random.randint(0, len(prev2))))
                    elif leaf == '翻转(镜像)' and j == 1:
                        dp_elements = copy.deepcopy(history_for_option[-1])
                    else:
                        temp_cfg = copy.copy(self.cfg)
                        setattr(temp_cfg, 'seed', distractor_seed)
                        res_opt = handler(history_for_option, config=temp_cfg, use_grid=use_grid)
                        dp_elements, dp_rule_info = res_opt if isinstance(res_opt, tuple) else (res_opt, None)
                    
                    if not dp_elements:
                        logger.warning(f"干扰项 {j} 处理器返回空元素")

                    # 检查是否与已有选项重复
                    current_set = self._elements_set(dp_elements)
                    existing_sets = [self._elements_set(opt['elements']) for opt in candidates_internal]
                    if current_set not in existing_sets:
                        break  # 找到有效干扰项

                except Exception as e_opt:
                    tbopt = traceback.format_exc()
                    logger.warning(f"干扰项 {j} 生成失败 (重试 {retry_count}): {str(e_opt)}\n{tbopt}")
                    dp_rule_info = {
                        'error': str(e_opt),
                        'traceback': tbopt,
                        'fallback': True,
                        'handler': handler_name
                    }

                retry_count += 1
                distractor_seed += 100  # 改变种子重试

            # 处理生成失败的情况
            if dp_elements is None:
                logger.warning(f"干扰项 {j} 达到最大重试次数，使用默认值")
                dp_elements = []  # 空元素作为fallback

            # 渲染干扰项
            try:
                dp_img = self._render_elements_to_canvas(W, H, self.cfg.bg_color, dp_elements, use_grid)
            except Exception as e_render:
                tb2 = traceback.format_exc()
                logger.error(f"干扰项 {j} 渲染失败: {str(e_render)}\n{tb2}")
                dp_img = np.ones((H, W, 3), dtype=np.uint8) * 255  # 白色图像作为fallback
                dp_rule_info = dp_rule_info or {}
                dp_rule_info['render_error'] = {'err': str(e_render), 'traceback': tb2}

            # 保存干扰项
            opt_path = os.path.join(sample_dir, f'option_{j}.png')
            save_image(dp_img, opt_path)
            candidates_internal.append({
                'img': dp_img,
                'path': opt_path,
                'is_correct': False,
                'elements': copy.deepcopy(dp_elements),
                'rule_info': dp_rule_info
            })
            option_states.append(dp_elements)

        # 打乱选项顺序
        if getattr(self.cfg, 'shuffle_options', False):
            local_rng.shuffle(candidates_internal)

        return candidates_internal

    def _compose_and_save_grid(self, sample_id: int, W: int, H: int, 
                              states_internal: List[Dict], candidates_internal: List[Dict],
                              sample_dir: str, num_options: int, show_labels: bool, 
                              show_border: bool) -> Tuple[str, Dict]:
        """组合网格图像并保存到单独目录"""
        logger.info(f"组合样本 {sample_id} 的网格图像")
        grid_im, cells_meta, seq_meta, opts_meta, query_path, grid_h, cell_size = compose_grid(
            W=W, H=H,
            states=states_internal[:-1],  # 排除最后一帧作为查询
            candidates=candidates_internal,
            sample_dir=sample_dir,
            num_options=num_options,
            margin=20,
            padding_v=20,
            show_labels=show_labels,
            show_border=show_border,
            bg_color=self.cfg.bg_color
        )

        # 保存grid图像到单独目录
        grid_path = os.path.join(self.grids_dir, f'grid_{sample_id:06d}.png')
        save_image(grid_im, grid_path)

        # 找到正确选项索引
        correct_index = next((i for i, c in enumerate(candidates_internal) if c['is_correct']), 0)

        return grid_path, {
            'correct_index': correct_index,
            'cells_meta': cells_meta,
            'seq_meta': seq_meta,
            'opts_meta': opts_meta,
            'grid_h': grid_h,
            'cell_size': cell_size
        }

    def _generate_metadata(self, sample_id: int, category_path: List[str], sample_dir: str,
                          grid_path: str, states_internal: List[Dict], 
                          candidates_internal: List[Dict], correct_index: int,
                          leaf: str, seed_base: int, grid_meta: Dict) -> Dict:
        """生成并保存元数据"""
        # 序列化序列状态元数据
        sequence_meta = [{
            'state_path': s['state_path'],
            'elements': self._serialize_elements_list(s['elements']),
            'canvas_size': list(s['canvas_size']),
            'rule_info': s['rule_info'],
            'timestamp': s['timestamp']
        } for s in states_internal]

        # 序列化选项元数据
        options_meta = [{
            'option_path': c['path'],
            'is_correct': c['is_correct'],
            'elements': self._serialize_elements_list(c['elements']),
            'rule_info': c['rule_info']
        } for c in candidates_internal]

        # 构建完整元数据
        meta = {
            'id': sample_id,
            'category_path': category_path,
            'sample_dir': sample_dir,
            'grid_path': grid_path,
            'sequence': sequence_meta,
            'options': options_meta,
            'correct_index': correct_index,
            'rule': leaf,
            'cells_meta': grid_meta['cells_meta'],
            'seed_info': {
                'cfg_seed': self.cfg.seed,
                'sample_seed': seed_base
            },
            'generation_time': datetime.utcnow().isoformat()
        }

        # 保存元数据
        if getattr(self.cfg, 'export_json', False):
            meta_path = os.path.join(sample_dir, 'meta.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logger.info(f"元数据已保存到 {meta_path}")

        # 导出COCO格式标注
        if getattr(self.cfg, 'export_coco', False):
            coco = {
                'images': [{
                    'id': sample_id,
                    'file_name': os.path.relpath(grid_path, self.out_dir),
                    'width': grid_meta['grid_h'],
                    'height': grid_meta['grid_h']
                }],
                'annotations': [],
                'categories': [{'id': 1, 'name': leaf}]
            }
            ann_id = 1
            for cell in grid_meta['cells_meta']:
                coco['annotations'].append({
                    'id': ann_id,
                    'image_id': sample_id,
                    'category_id': 1,
                    'bbox': cell['bbox'],
                    'label': cell.get('label', '')
                })
                ann_id += 1
            coco_path = os.path.join(sample_dir, 'coco.json')
            with open(coco_path, 'w', encoding='utf-8') as f:
                json.dump(coco, f, ensure_ascii=False, indent=2)
            logger.info(f"COCO标注已保存到 {coco_path}")

        # 清理内存
        for s in states_internal:
            s['state_img'] = None
        for c in candidates_internal:
            c['img'] = None

        return meta

    # ---------- 类别采样方法 ----------
    def _sample_category_leaf(self) -> List[str]:
        """从配置的类别中采样叶子节点"""
        leaves = []
        
        def traverse_category(d: Any, path: List[str]) -> None:
            if isinstance(d, dict):
                for k, v in d.items():
                    traverse_category(v, path + [k])
            elif isinstance(d, list):
                for item in d:
                    leaves.append(path + [item])
        
        traverse_category(self.cfg.categories, [])
        weights = [self.cfg.category_weights.get(l[-1], 1.0) for l in leaves]
        total_weight = sum(weights)
        return random.choices(leaves, weights=weights if total_weight > 0 else None, k=1)[0]