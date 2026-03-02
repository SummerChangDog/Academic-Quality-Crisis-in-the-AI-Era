import os
import json
import csv
# ===================== 配置项（请根据你的实际路径修改） =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 文件夹
RESULTS_FOLDER = os.path.join(BASE_DIR, "../results")
# 生成的CSV文件保存路径和名称
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "../results/paper_rates_new.csv")

# 要提取的字段（按你要求的列顺序）
FIELDS = [
    "arxiv_id",
    "domain",
    "year",
    "empirical_clarity_score",
    "explanation_vs_speculation_score",
    "language_misuse_score",
    "math_quality_score",
    "citation_ai_rate",
    "academic",
    "industry"
]

def main():
    # 存储所有提取的数据
    all_data = []
    
    # 遍历results文件夹下的所有文件（包括子文件夹）
    for root, dirs, files in os.walk(RESULTS_FOLDER):
        for filename in files:
            # 筛选文件名包含"eval_results_"的文件（匹配你描述的文件名格式）
            if "eval_results_" in filename:
                file_path = os.path.join(root, filename)
                try:
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 解析JSON内容
                        file_content = json.load(f)
                    
                    # 提取指定字段，确保字段缺失时用None填充
                    row_data = {}
                    for field in FIELDS:
                        row_data[field] = file_content.get(field, None)
                    
                    # # ===================== 新增：调用函数获取机构类型 =====================
                    # arxiv_id = row_data.get("arxiv_id")
                    # if arxiv_id:  # 仅当有arxiv_id时才调用函数
                    #     row_data["institution"] = get_institution_type(arxiv_id)
                    #     print(arxiv_id,":ok!")
                    # else:
                    #     row_data["institution"] = None
                    #     print(f"⚠️  文件 {file_path} 缺少arxiv_id，跳过机构类型识别")

                    # 添加到数据列表
                    all_data.append(row_data)
                    print(f"成功读取并解析文件: {file_path}")
                
                except json.JSONDecodeError:
                    print(f"⚠️  警告：文件 {file_path} 不是有效的JSON格式，已跳过")
                except FileNotFoundError:
                    print(f"⚠️  警告：文件 {file_path} 不存在，已跳过")
                except PermissionError:
                    print(f"⚠️  警告：没有权限访问文件 {file_path}，已跳过")
                except Exception as e:
                    print(f"⚠️  警告：处理文件 {file_path} 时出错: {str(e)}，已跳过")
    
    # 写入CSV文件
    if all_data:
        with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            # 创建CSV写入器
            writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
            # 写入列名
            writer.writeheader()
            # 写入所有数据行
            writer.writerows(all_data)
        
        print(f"\n✅ 处理完成！共解析 {len(all_data)} 个文件")
        print(f"📄 生成的CSV文件已保存至：{os.path.abspath(OUTPUT_CSV_PATH)}")
    else:
        print("\n❌ 未找到任何符合条件的文件，请检查文件夹路径或文件格式")

if __name__ == "__main__":
    # 检查配置的文件夹是否存在
    if not os.path.exists(RESULTS_FOLDER):
        print(f"❌ 错误：配置的results文件夹路径不存在 -> {RESULTS_FOLDER}")
        print("请修改代码中的 RESULTS_FOLDER 为正确的绝对路径")
    else:
        main()
        