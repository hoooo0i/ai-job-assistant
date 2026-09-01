# AI 求职助手 MVP

这是一个可本地运行的 Streamlit MVP，当前实现：

- 上传一份 PDF 简历并使用 PyMuPDF 在内存中提取文字；
- PDF 无法提取有效文字时，显示简历文本粘贴框作为备用入口；
- 输入公司、岗位、地点、岗位类型和岗位 JD；
- 校验文件与文本，并展示经过隐私脱敏的输入预览；
- 使用 GPT-5.5 和 OpenAI Responses API Structured Outputs，把简历与 JD 分别解析为 Pydantic 结构；
- 针对每项岗位要求输出 `matched`、`partial`、`missing` 或 `unknown`，并显示可验证的简历原文证据；
- 由 Python 规则引擎计算“证据匹配度”和“信息完整度”；
- 生成与具体 JD 要求关联的简历优化建议、待确认问题和面试准备问题。
- 用户可以回答 `unknown` 硬性条件，系统无需再次调用模型即可更新状态并重新计算两项分数。

本阶段不包含 RAG、录取概率预测、数据库或模型训练。

## 评分规则

- 重要程度权重：`must_have=3`、`preferred=2`、`other=1`。
- 状态系数：`matched=1`、`partial=0.5`、`missing=0`。
- `unknown` 不进入证据匹配度分母，但会降低信息完整度。
- 用户对待确认条件选择“符合”或“不符合”后，状态分别更新为 `matched` 或 `missing`，置信度记为用户明确确认。
- 证据匹配度不是录取概率或 ATS 通过率。

## 输入规则

- PDF 必须上传，最大 10 MB。
- Streamlit 上传控件通过 `.streamlit/config.toml` 同步限制为 10 MB。
- PDF 或备用粘贴文本至少包含 50 个非空白字符。
- 只有 PDF 无法提取有效文字时才显示粘贴框。
- 文本来源优先级固定为：有效 PDF 文本 > 用户粘贴文本；两者不会拼接。
- 公司名称和岗位名称必填；JD 至少包含 50 个非空白字符。

## 隐私说明

应用不会主动把原始 PDF、提取文字或粘贴内容写入磁盘、数据库或日志。数据仅存在于当前 Streamlit 会话内。简历预览和发送给模型的简历文本会遮盖常见的电话、邮箱和详细地址。OpenAI API 请求设置 `store=False`。

分析结果和待确认回答只保存在当前 Streamlit 会话状态中；更换 PDF、JD 或岗位信息会自动清除旧结果。

简历优化建议只允许重组或强化已有事实。程序会过滤无法定位到简历原文、关联无效岗位要求或新增简历中不存在数字的建议；缺少量化信息时应改为向用户提问。

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
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── ai_parser.py
│   ├── matching.py
│   ├── pdf_parser.py
│   ├── schemas.py
│   ├── validators.py
│   └── privacy.py
└── tests/
    ├── test_ai_parser.py
    ├── test_matching.py
    ├── test_pdf_parser.py
    ├── test_validators.py
    └── test_privacy.py
```
