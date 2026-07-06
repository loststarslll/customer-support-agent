# 智能客服 Agent 系统

本项目是一个面向电商客服场景的智能客服 Agent 原型系统，基于 DeepSeek、FAISS、RAG、Skill、Tool Calling、FastAPI 和 Streamlit 构建。

系统支持 FAQ 知识库问答、订单状态查询、多轮参数补全、订单权限校验、Agent 执行轨迹记录、SQLite 日志存储、自动评测、API 服务化以及用户端/商家端可视化界面。

---

## 一、项目目标

本项目旨在构建一个能够处理常见电商客服问题的智能客服系统。

主要目标包括：

- 支持用户用自然语言咨询客服问题；
- 使用 RAG 从 FAQ 知识库中检索相关内容并生成回答；
- 使用 Tool Calling 查询订单、物流和退款状态；
- 支持多轮对话中的订单号补全；
- 在订单查询中加入 `user_id` 权限校验；
- 使用 Skill 机制封装不同客服能力；
- 记录 Agent 执行轨迹和请求日志；
- 提供 FastAPI 后端接口；
- 提供 Streamlit 用户端和商家管理端页面；
- 通过自动评测验证系统效果。

---

## 二、项目技术栈

| 模块 | 技术 |
|---|---|
| 大语言模型 | DeepSeek API |
| RAG 框架 | LangChain |
| 向量数据库 | FAISS |
| Embedding 模型 | sentence-transformers 多语言模型 |
| Agent 控制 | Planner / Controller / Reviewer |
| 能力封装 | Skill Registry |
| 工具调用 | Order Tool |
| 后端服务 | FastAPI |
| 前端页面 | Streamlit |
| 日志存储 | SQLite |
| 配置管理 | python-dotenv |
| 自动评测 | Python 测试脚本 |

---

## 三、系统整体架构

```text
用户
→ Streamlit 用户端
→ FastAPI 后端接口
→ Agent Pipeline
→ Planner 判断意图
→ Controller 调度执行
→ Skill Registry 选择能力
→ Skill 执行业务流程
   ├── FAQSkill
   │   └── FAISS 检索 + LLM 回答
   │
   └── OrderQuerySkill
       └── query_order Tool + 订单权限校验
→ Reviewer 检查 Observation
→ 返回回答
→ SQLite 记录日志

## 一键启动

```bash
bash scripts/start_all.sh

---

## 数据与评测链路

本项目采用“公开 FAQ → 中文知识库 → 中文口语评测集 → 检索指标 → 用户端引用溯源”的数据闭环。

知识库基于 Hugging Face 公开客服 FAQ 数据集构建，包括 `MakTek/Customer_support_faqs_dataset` 和 `Andyrasika/Ecommerce_FAQ`。两个数据集原始样本共 279 条，经 question-answer 去重后得到 89 条唯一 FAQ。系统将英文 FAQ 翻译为中文知识库，并保留 `question_en`、`answer_en`、`dataset_source`、`source_title` 等字段用于来源追溯。

中文检索评测集从公开 FAQ 中抽样生成，并进一步改写为口语化中文用户问题，以更贴近真实客服场景。评测指标分为严格 FAQ ID 指标和弱标注 Category 指标，包括 FAQ ID Recall@3、FAQ ID MRR、Category Precision@3 等。

用户端回答会展示 FAQ 参考来源，例如 `[public_zh_faq_0007] 如何申请退款？｜MakTek Customer Support FAQs Dataset｜2026-07-01`，用于验证回答的可追溯性。

