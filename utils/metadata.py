from habanero import Crossref
import json
import requests
from typing import Dict, List, Optional
from utils.api import chat

HEADERS = {
    "User-Agent": "CitationChecker/1.0 (mailto:csy24@ruc.edu.cn)",
    "Accept": "application/json"
}

def search_by_title_author(title: str, authors: List[str]) -> Optional[Dict]:
    """通过标题+作者搜索CrossRef，返回最匹配的元数据"""
    if not title or not authors:
        return None
    # 构造搜索关键词（作者姓氏 + 标题核心词）
    author_surnames = [a.split()[-1] for a in authors if a.strip()]
    query = f"{', '.join(author_surnames)} {title[:100]}"  # 截断过长标题

    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": 1,
        "sort": "relevance"
    }

    try:
        # 直接使用 requests 调用 API
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            items = response.json().get("message", {}).get("items", [])
        if items:
            return items[0]  # 返回最相关的结果
        return None
    except Exception as e:
        print(f"标题+作者搜索失败 {title[:50]}: {str(e)}")
        return None

def parse_citation_text(cite_dict: Dict) -> Dict:
    """
    解析规整的引用字典，提取结构化数据（适配你的输入格式）
    输入：包含index/id/text/links/doi/arxiv_id的引用字典
    输出：保持原结构的parsed字典（兼容后续验证逻辑）
    """
    parsed = {
        "id": "",         
        "authors": [],     
        "title": "",       
        "year": None,     
        "doi": None,       
        "journal": "",     
        "original_text": ""
    }

    # 1. 直接提取已有结构化字段
    parsed["id"] = cite_dict.get("id", "").strip()       
    parsed["doi"] = cite_dict.get("doi")                 
    parsed["original_text"] = cite_dict.get("text", "").strip() 
    cite_text = parsed["original_text"]  
    # 2. 轻量化解析text中的核心信息
    if not cite_text:
        return parsed  

    escaped_cite_text = json.dumps(cite_text, ensure_ascii=False)[1:-1]  # 去掉首尾的双引号
    # 构造精准的prompt，要求大模型返回指定格式的JSON
    prompt = f"""
请严格按照以下要求解析引文文本，返回JSON格式的结果，不要添加任何额外解释或文本：
1. 提取字段说明：
   - authors：作者列表（字符串数组，每个元素是单个作者的完整姓名，如["N. Carlini", "S. Chien"]）
   - title：论文标题（字符串，去除末尾的标点符号）
   - year：年份（整数，仅提取4位数字，如2022a提取2022）
   - journal：期刊/会议名称（字符串，去除多余标点和空格）
2. 输入引文文本：{escaped_cite_text}
3. 返回格式示例（必须严格遵循）：
{{
    "authors": ["N. Carlini", "S. Chien", "M. Nasr", "S. Song", "A. Terzis", "F. Tramer"],
    "title": "Membership inference attacks from first principles",
    "year": 2022,
    "journal": "2022 IEEE symposium on security and privacy (SP)"
}}
千万注意，必须只返回3中示例的部分，也就是{...}，不需要加任何自然语言解释/markdown代码块！！！
例如，你的输出格式不应该是以下的：
```json
{{
    "authors": ["A. M. Turing"],
    "title": "Computing Machinery and Intelligence",
    "year": 1950,
    "journal": "Mind"
}}
```
而应该是模仿以下的：
{{
    "authors": ["A. M. Turing"],
    "title": "Computing Machinery and Intelligence",
    "year": 1950,
    "journal": "Mind"
}}
以上只是示例，你不要直接照抄上面的输出！
"""
    # 调用大模型获取解析结果
    llm_response = chat(prompt, model="gemini-2.5-flash")
    print(llm_response)
    try:
        # 解析大模型返回的JSON字符串
        llm_parsed = json.loads(llm_response.strip())
    except:
        return None
    
    # 填充解析结果到parsed字典（做类型校验，避免格式错误）
    parsed["authors"] = llm_parsed.get("authors", []) if isinstance(llm_parsed.get("authors"), list) else []
    parsed["title"] = llm_parsed.get("title", "").strip() if isinstance(llm_parsed.get("title"), str) else ""
    parsed["year"] = llm_parsed.get("year") if isinstance(llm_parsed.get("year"), int) else None
    parsed["journal"] = llm_parsed.get("journal", "").strip() if isinstance(llm_parsed.get("journal"), str) else ""

    return parsed