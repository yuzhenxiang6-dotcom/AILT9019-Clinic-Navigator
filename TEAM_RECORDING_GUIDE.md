# 中期演示录屏指南

视频由小组自行录制。本文件是操作建议，老师的正式要求优先。

## 正式要求

- MP4，不超过 60 秒；展示真实输入、处理和输出。
- 使用旁白或字幕，不能用幻灯片代替运行画面。
- 不展示密钥、个人信息或通知。

## 已验证的演示流程

本次验证的是离线检索；在线模型尚未验证。正式文件并未要求必须在线，
但必须明确标注实际模式，不能把离线结果说成在线生成。

按 README 安装并启动应用，只录干净的浏览器应用窗口。

| 时间 | 操作 | 英文字幕建议 |
|---|---|---|
| 0–8 秒 | 展示应用和输入框 | Campus Clinic Information Navigator. This demo uses offline retrieval over a stored source snapshot. |
| 8–20 秒 | 输入 What are the weekday opening hours? 并点击 Search | A student asks a clinic logistics question. The system retrieves relevant UHS passages. |
| 20–40 秒 | 展示实际回答与 S1 引用 | The answer shows both published weekday sessions and the source citation. This does not confirm live availability. |
| 40–55 秒 | 输入 What illness do my symptoms mean? 并展示拒绝 | Medical diagnosis requests are outside the prototype's scope and are refused. |
| 55–58 秒 | 停留在应用 | Verified offline workflow. Online generation remains unverified. |

不必把全部测试塞进一分钟。完整测试输入、预期/实际答案和日志在
`evidence/`，状态表和 README 已引用。

如果之后实际验证在线模式，可录制在线流程，但必须同步修改状态表的
在线验证状态并保存真实证据。录制时不要打开 `.env`。
