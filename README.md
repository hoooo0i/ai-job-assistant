# AI 求职与面试助手 MVP

这是一个可本地运行的 Streamlit MVP，当前实现：

- 上传一份 PDF 简历并使用 PyMuPDF 在内存中提取文字；
- PDF 无法提取有效文字时，显示简历文本粘贴框作为备用入口；
- 输入公司、岗位、地点、岗位类型和岗位 JD，或从公开招聘链接自动读取并回填；
- 首次可一次提交最多 5 份 JD，完成后进入岗位横向对比工作台；
- 在同一简历会话中继续补充岗位，并点击任一岗位进入独立的详细分析流程；
- 将多岗位排序导出为经过脱敏的 Word 或 PDF 对比报告；
- 手动导出/导入脱敏 JSON 求职档案，恢复岗位分析、用户确认事实和简历版本记录；
- 为每个岗位记录准备、投递、笔试、面试、拒绝、Offer 或放弃状态；
- 绑定实际使用的简历版本，记录岗位链接、投递日期、截止日期、面试日期、跟进日期和备注；
- 在岗位工作台展示投递数、回复率、面试率、Offer 数以及未来 30 天应用内提醒；
- 校验文件与文本，并展示经过隐私脱敏的输入预览；
- 使用 GPT-5.5 和 OpenAI Responses API Structured Outputs，把简历与 JD 分别解析为 Pydantic 结构；
- 按“上传资料 → 初步匹配 → 补充真实信息 → 最终材料”四步向导运行；
- 针对每项岗位要求输出 `matched`、`partial`、`missing` 或 `unknown`，并显示可验证的简历原文证据；
- 对部分匹配、缺失和待确认要求最多提出 5 个补充问题，补充页仅需选择“具备、不具备或不确定”；
- 初始匹配同时生成并暂存建议，提交补充选项时由 Python 本地更新，不再次调用模型；
- 由 Python 规则引擎计算“证据匹配度”和“信息完整度”；
- 生成与具体 JD 要求关联的简历优化建议、待确认问题和面试准备问题。
- 用户可以回答 `unknown` 硬性条件，系统无需再次调用模型即可更新状态并重新计算两项分数。
- 按已覆盖、待强化、能力缺口和待确认分类展示 JD 关键要求。
- 提供简历优化工作台和高优先级补充卡，用户可在这里补充情境、行动、结果及真实数据，并用一次模型调用批量优化多项内容。
- 未完成的占位符不会进入定制 Word 简历，用户可逐条采纳、编辑或忽略建议。
- 每个岗位可保存最多 10 个定制简历版本，并支持下载、恢复到编辑器和删除。
- 使用本地规则检查 PDF 可解析性、页数、多栏、表格、图片、字号、标准板块、日期格式和 JD 关键词覆盖。
- 在简历优化中高亮显示修改前后的新增与删除内容。
- 下载定制简历前检查占位符、无来源数字、目标岗位和未处理建议，并要求用户确认事实准确。
- 最终结果使用会话有状态的功能导航；保存、生成或校验报错后仍停留在当前功能，不会跳回“岗位匹配”。
- 生成中文或英文求职信，每段必须绑定简历证据，支持 Word/PDF 下载。
- 按简历证据生成面试回答思路和 STAR 框架；没有证据时不生成第一人称经历。
- 提供文字模拟面试，对完整性、STAR、岗位相关性和表达清晰度进行点评。
- 在内存中生成经过脱敏的 Word 和 PDF 报告，并提供下载和建议复制入口。
- 将岗位报告、投递检查清单、已保存的定制简历及已生成的求职信整理成一个 ZIP 投递材料包。
- 通过统一模型适配层调用 OpenAI，为后续接入本地模型预留边界。

本阶段不包含 RAG、语音监听、录取概率预测、数据库或模型训练。

## 评分规则

- 重要程度权重：`must_have=3`、`preferred=2`、`other=1`。
- 状态系数：`matched=1`、`partial=0.5`、`missing=0`。
- `unknown` 不进入证据匹配度分母，但会降低信息完整度。
- 用户只确认技能或经历但未提供细节时按 `partial` 计分；时间、地点等可直接确认的条件可更新为 `matched`。
- 选择“不具备”更新为 `missing`，选择“不确定”更新为 `unknown`，未回答保持初步判断。
- 证据匹配度不是录取概率或 ATS 通过率。

## 输入规则

- PDF 必须上传，最大 10 MB。
- Streamlit 上传控件通过 `.streamlit/config.toml` 同步限制为 10 MB。
- PDF 或备用粘贴文本至少包含 50 个非空白字符。
- 只有 PDF 无法提取有效文字时才显示粘贴框。
- 文本来源优先级固定为：有效 PDF 文本 > 用户粘贴文本；两者不会拼接。
- 公司名称和岗位名称必填；JD 至少包含 50 个非空白字符。
- 岗位链接是选填项。链接读取优先识别网页中的 `JobPosting` 结构化数据，失败时尝试正文；登录页、纯动态页面和限制抓取的页面仍需手动粘贴 JD。
- 为防止访问本机服务或下载超大页面，链接导入只接受公网 HTTP/HTTPS 地址，最多跟随 4 次跳转，页面上限为 2 MB。

## 隐私说明

应用不会主动把原始 PDF、提取文字或粘贴内容写入磁盘、数据库或日志。数据仅存在于当前 Streamlit 会话内。简历预览和发送给模型的简历文本会遮盖常见的电话、邮箱和详细地址。OpenAI API 请求设置 `store=False`。

默认情况下，候选人档案、用户补充事实、岗位分析和模拟面试反馈只保存在当前 Streamlit 会话状态中。用户也可以主动下载脱敏 JSON 档案，并在之后手动导入恢复；应用不会自动上传或长期保存该档案。档案不包含原始 PDF、Word/PDF 二进制缓存、电话、邮箱或详细地址。简历原文证据标记为“简历”，用户填写的内容标记为“用户确认”，两者不会混淆。

会话状态将候选人档案与岗位分析分开，并按岗位 ID 隔离结果。工作台可横向比较最多 5 个岗位，进入详情后一次只展示一个岗位。

岗位对比工作台使用本地规则汇总匹配度、完整度、ATS 分数、硬性风险、必须项缺口和推荐值。推荐值只用于整理投递顺序，不代表录取概率；每个岗位的补充回答和生成材料互不覆盖。

定制简历和求职信同样只在当前会话内生成。定制简历是可编辑内容草稿，不保留原 PDF 排版和联系方式；用户需在投递前核对事实并自行补充。

投递材料包同样只在内存中生成，不包含用户上传的原始 PDF。若尚未保存定制简历版本或生成求职信，ZIP 会保留分析报告和检查清单，并明确提示缺失的可选材料。

PDF 报告使用项目内置的 Noto Sans SC 字体保证中文显示，字体按 SIL Open Font License 授权，许可证见 `assets/fonts/OFL.txt`。

简历优化建议只允许重组或强化已有事实。程序会过滤无法定位到简历原文、关联无效岗位要求或新增简历中不存在数字的建议；缺少量化信息时应改为向用户提问。

## 模型调用与会话缓存

- 首次完成主流程通常调用 3 次模型：解析简历、解析 JD、一次生成匹配及后续建议。
- 同一批次及后续新增岗位会复用简历解析；每个未缓存的新 JD 通常增加 2 次调用。
- 打开岗位、返回对比和本地排序不会调用模型。
- 保存投递记录、更新统计和日期提醒不调用模型。
- 从岗位链接读取 JD、生成报告及整理 ZIP 材料包不调用模型。
- 提交、跳过或修改补充选项不调用模型，状态与分数由 Python 本地更新。
- “AI 批量优化”、求职信和面试工具仅在用户主动点击时调用，并在页面显示当前会话累计次数。
- 相同简历、JD、提供方和模型在当前会话内复用解析缓存；更换内容或模型会自动使用新的缓存键。

请勿在公开演示、README、测试数据或作品集材料中放入真实简历和联系方式。

## 本地运行

要求 Python 3.11 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY
streamlit run app.py
```

默认访问地址：<http://localhost:8501>

## 运行测试

```bash
source .venv/bin/activate
pytest -q
```

测试使用程序生成的合成 PDF 和虚构文本，不需要真实简历或岗位数据。

## 项目结构

```text
ai-job-assistant/
├── app.py
├── assets/
│   └── fonts/
│       ├── NotoSansSC.ttf
│       └── OFL.txt
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── ai_parser.py
│   ├── ai_provider.py
│   ├── application_package.py
│   ├── application_tracker.py
│   ├── archive.py
│   ├── ats_checker.py
│   ├── career_tools.py
│   ├── comparison.py
│   ├── evidence_flow.py
│   ├── interview.py
│   ├── job_link.py
│   ├── matching.py
│   ├── pdf_parser.py
│   ├── reporting.py
│   ├── resume_versions.py
│   ├── schemas.py
│   ├── submission.py
│   ├── validators.py
│   └── privacy.py
└── tests/
    ├── test_ai_parser.py
    ├── test_ai_provider.py
    ├── test_application_package.py
    ├── test_application_tracker.py
    ├── test_archive.py
    ├── test_ats_checker.py
    ├── test_app.py
    ├── test_career_tools.py
    ├── test_comparison.py
    ├── test_evidence_flow.py
    ├── test_interview.py
    ├── test_job_link.py
    ├── test_matching.py
    ├── test_pdf_parser.py
    ├── test_validators.py
    ├── test_privacy.py
    ├── test_reporting.py
    ├── test_resume_versions.py
    └── test_submission.py
```

## 模型配置

- `AI_PROVIDER=openai`：当前已实现的模型提供方。
- `AI_MODEL=gpt-5.5`：使用的模型名称。
- 仍兼容旧的 `OPENAI_MODEL` 配置；当 `AI_MODEL` 未填写时自动回退。
- 后续接入 Ollama 或 vLLM 时，只需新增相同结构化输出接口的适配器，不改动页面和评分规则。
