from utils.metadata import search_by_title_author, parse_citation_text
from itertools import islice
from utils.api import chat, extract_score
import os
import json
import argparse

prompt_template = """
Below is the beginning content of an academic paper. You have to distinguish whether the authors come from academic institutions (e.g. universitiy, college, research institute etc.) or industry (e.g. company, enterprise, etc.). 

You need to give the result wrapped in the pairs <academic> </academic> and <industry> </industry>.
If there exists any author from academic affiliations, put a 1 in the <academic> </academic> brackets, otherwise 0.
If there exists any author from industrial affiliations, put a 1 in the <industry> </industry> brackets, otherwise 0.

Example: 
<academic>1</academic>
<industry>0</industry>

Paper content:
{}
"""

def parse_affiliations(arxiv_id):
    path = f"data/papers/{arxiv_id}/body.txt"
    with open(path, "r", encoding="utf-8") as f:
        head = "".join(islice(f, 40))

    prompt = prompt_template.format(head)
    response = chat(prompt)
    return {
        "academic": extract_score(response, "academic"),
        "industry": extract_score(response, "industry")
    }
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Getting Affiliation.")
    ap.add_argument("year", type=int, help="Year, e.g., 23")
    args = ap.parse_args()
    year = str(args.year)
    print(year)
    for root, dirs, files in os.walk("results/", topdown=True, followlinks=False):
        for name in files:
            if name[-15:-13] != year:
                continue
            if "cs.ai" in name:
                arxiv_id = name[19:29]
            else:
                arxiv_id = name[18:28]
            print(arxiv_id)
            
            aff = parse_affiliations(arxiv_id)
            with open(f"results/{name}") as f:
                results = json.load(f)
            results["academic"] = aff["academic"]
            results["industry"] = aff["industry"]
            with open(f"results/{name}", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)