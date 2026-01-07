import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.patches import Patch

def _ensure_renderer(fig):
    # 强制 draw，确保 renderer 可用并且 artist window extents 已计算
    try:
        fig.canvas.draw()
    except Exception:
        # 某些 headless 环境可能会在 draw 时抛错，但大多数场景 draw 是必须的
        pass
    return fig.canvas.get_renderer()

def _display_bbox_from_artist(ax, artist, renderer):
    """
    返回 artist 在 display 坐标系下的 Bbox（matplotlib.transforms.Bbox）。
    尽可能使用 artist.get_window_extent(renderer)，失败时做类型特定的计算。
    可能抛异常以让调用方决定如何回退。
    """
    # 优先使用官方的 window extent（已包含 transform 与 stroke）
    try:
        disp_bbox = artist.get_window_extent(renderer)
        return disp_bbox
    except Exception:
        pass

    # Line2D：用 data points -> display transform
    if isinstance(artist, Line2D):
        x = np.asarray(artist.get_xdata())
        y = np.asarray(artist.get_ydata())
        if x.size == 0 or y.size == 0:
            raise RuntimeError("Line2D 没有数据点")
        pts = np.column_stack((x, y))
        disp_pts = ax.transData.transform(pts)
        xmin, ymin = float(disp_pts[:,0].min()), float(disp_pts[:,1].min())
        xmax, ymax = float(disp_pts[:,0].max()), float(disp_pts[:,1].max())
        return Bbox.from_extents(xmin, ymin, xmax, ymax)

    # PathCollection (scatter 等)：尝试 offsets -> data -> display
    if isinstance(artist, PathCollection):
        # 优先用 get_offsets，如果没有或为空抛出
        offsets = None
        try:
            offsets = artist.get_offsets()
        except Exception:
            offsets = None
        if offsets is None or len(offsets) == 0:
            raise RuntimeError("PathCollection 无 offsets 可用")
        arr = np.asarray(offsets)
        # offsets 可能已经在 data coord，也可能在其他 coord；一般情况下把它当作 data 并通过 ax.transData 转换是稳妥的回退方案
        disp_pts = ax.transData.transform(arr[:, :2])
        xmin, ymin = float(disp_pts[:,0].min()), float(disp_pts[:,1].min())
        xmax, ymax = float(disp_pts[:,0].max()), float(disp_pts[:,1].max())
        return Bbox.from_extents(xmin, ymin, xmax, ymax)

    # Patch 但没有 get_window_extent（极少）：尝试 path+transform -> 将其目标 transform 变换到 display
    if isinstance(artist, Patch):
        try:
            path = artist.get_path()
            bbox_path = path.get_extents()  # path local extents
            transform = artist.get_transform()
            # transform 是把 path 的点变换到 artist 的父坐标系（通常是 data 或 axes coords），
            # 但我们需要 display 坐标：把 transform 结果再乘以 ax.transData (如果 transform 到 data)
            # 最稳妥的方式就是先得到目标 bbox（transform 的目标坐标系），然后尝试把它通过适当 transform 映射到 display。
            bbox_target = transform.transform_bbox(bbox_path)
            # 试着把 bbox_target 转到 display：如果 bbox_target 是 data-space（常见情况），使用 ax.transData
            try:
                disp_bbox = ax.transData.transform_bbox(bbox_target)
                return disp_bbox
            except Exception:
                # 作为最后手段，直接把 bbox_target 当 display（不常见）
                return bbox_target
        except Exception:
            pass

    # 如果到这里还没返回，抛异常让调用方记录为无法计算
    raise RuntimeError(f"无法为 artist 计算 display bbox: {type(artist)}")

def check_axes_artists_inside(ax: plt.Axes, tol: float = 1e-9):
    """
    统一在 display 坐标系下检查 artists 是否在 axes 可见范围内。
    返回 dict，out_of_bounds 中包含 artist、reason、bbox_display、bbox_data（若能计算）。
    """
    fig = ax.figure
    renderer = _ensure_renderer(fig)
    ax_disp_bbox = ax.get_window_extent(renderer)  # axes 在 display 空间的 bbox

    out = []
    checked = 0

    # 遍历 patches, lines, collections
    artists = []
    artists.extend(list(ax.patches))
    artists.extend(list(ax.lines))
    artists.extend(list(ax.collections))

    for art in artists:
        try:
            # 1) 得到 display bbox（优先方法）
            try:
                art_disp_bbox = _display_bbox_from_artist(ax, art, renderer)
            except Exception as e:
                out.append({"artist": art, "bbox_display": None, "bbox_data": None,
                            "reason": f"无法计算 display bbox: {e}"})
                continue

            # 2) 判断 display bbox 是否完全位于 ax display bbox 内
            fully_inside = (
                art_disp_bbox.x0 + tol >= ax_disp_bbox.x0 and
                art_disp_bbox.x1 - tol <= ax_disp_bbox.x1 and
                art_disp_bbox.y0 + tol >= ax_disp_bbox.y0 and
                art_disp_bbox.y1 - tol <= ax_disp_bbox.y1
            )

            # 3) 取得 data bbox（用于日志）
            try:
                data_bbox = ax.transData.inverted().transform_bbox(art_disp_bbox)
            except Exception:
                data_bbox = None

            checked += 1
            if not fully_inside:
                out.append({
                    "artist": art,
                    "reason": "部分或全部超出画布范围",
                    "bbox_display": art_disp_bbox,
                    "bbox_data": data_bbox
                })
        except Exception as e:
            out.append({"artist": art, "bbox_display": None, "bbox_data": None,
                        "reason": f"检查时出错: {e}"})

    return {
        "all_inside": len(out) == 0,
        "out_of_bounds": out,
        "checked_count": checked
    }