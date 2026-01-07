import http.client
import json

conn = http.client.HTTPSConnection("yunwu.ai")
payload = json.dumps({
   "systemInstruction": {
      "parts": [
         {
            "text": "你是一直小猪.你会在回复开始的时候 加一个'哼哼'"
         }
      ]
   },
   "contents": [
      {
         "role": "user",
         "parts": [
            {
               "text": "你是谁?"
            }
         ]
      }
   ],
   "generationConfig": {
      "temperature": 1,
      "topP": 1,
      "thinkingConfig": {
         "includeThoughts": True,
         "thinkingBudget": 26240
      }
   }
})
headers = {
   'x-goog-api-key': 'sk-MLvhPPQVCgBrWzx7CHvDVGcgssBUpsYxLsolUmNFDYII1b5N',
   'Content-Type': 'application/json'
}
conn.request("POST", "/v1beta/models/gemini-2.5-pro:generateContent", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))