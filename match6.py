import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import re


def Label_processing(merge_df):
    """标签处理"""


def label_merge(merge_df):
    """标签转换"""
    result_df = merge_df.copy()
    
    # 从特殊项目标签提取标签
    def extract_special_project_tags(row):
        if pd.isna(row['特殊项目标签']):
            return pd.Series({
                '博士成功案例': '', 
                '低龄留学成功案例': '', 
            })
        
        tags = str(row['特殊项目标签']).split('、')
        result = {
            '博士成功案例': [],
            '低龄留学成功案例': [],
        }
        
        # 检查每个标签是否包含关键词
        for tag in tags:
            if '博士成功案例' in tag:
                result['博士成功案例'].append(tag)
            if '低龄留学成功案例' in tag:
                result['低龄留学成功案例'].append(tag)
        
        # 将列表转换为顿号分隔的字符串
        return pd.Series({
            k: '、'.join(v) if v else '' for k, v in result.items()
        })
    
    # 提取标签

    special_project_tags = result_df.apply(extract_special_project_tags, axis=1)
    
    # 合并所有标签
    result_df = pd.concat([
        result_df['文案顾问业务单位'],
        result_df['国家标签'],
        result_df['专业标签'],
        result_df['名校专家'],
        special_project_tags,
        result_df['行业经验'],
        result_df['文案背景'],
        result_df['业务单位所在地']
    ], axis=1)
    
    # 确保列的顺序正确
    desired_columns = [
        "文案顾问业务单位", "国家标签", "专业标签", 
        "名校专家", "博士成功案例", "低龄留学成功案例", 
        "行业经验", "文案背景", "业务单位所在地",
    ]
    
    # 确保所有列都存在，如果不存在则添加空列
    for col in desired_columns:
        if col not in result_df.columns:
            result_df[col] = ''
    
    # 按照期望的顺序重排列
    result_df = result_df[desired_columns]
    
    return result_df

def Consultant_matching(consultant_tags_file, merge_df):
    """顾问匹配"""
        
    # 定义标签权重
    global tag_weights
    tag_weights = {
        '绝对高频国家': 35,
        '相对高频国家': 20,
        '做过国家':5,
        '绝对高频专业': 15,
        '相对高频专业': 10,
        '做过专业':5,
        '名校专家': 10,
        '博士成功案例': 10,
        '低龄留学成功案例': 10,
        '行业经验': 15,
        '文案背景': 10,
        '业务单位所在地': 5
    }
        
        
    # 定义工作量权重
    global workload_weights
    workload_weights = {
        '学年负荷': 25,
        '近两周负荷': 25,
        '文书完成率': 25,
        '申请完成率': 25
    }
        
    # 定义个人意愿权重
    global personal_weights
    personal_weights = {
        '个人意愿': 100,
    }
        
    # 定义评分维度权重
    global dimension_weights
    dimension_weights = {
        '标签匹配': 0.5,
        '工作量': 0.3,
        '个人意愿': 0.2
    }

    def calculate_tag_matching_score(case, consultant, direction):
        """计算标签匹配得分"""
        tag_score_dict = {}
        
        # 1. 国家标签匹配
        if '国家标签' in case and pd.notna(case['国家标签']):
            # 处理案例国家
            raw_case_countries = case['国家标签']
            split_case_countries = re.split(r'[、,，\s]+', raw_case_countries)
            case_countries = {country.strip() for country in split_case_countries}
            
            # 计算加权后的总国家数
            weighted_total = sum(2 if country == '加拿大' else 1 for country in case_countries)
            
            if "美国" in case_countries and consultant['文案方向'] != '美国':
                direction = False
                return tag_score_dict, direction
            
            # 处理顾问国家
            raw_absolute = consultant['绝对高频国家'] if pd.notna(consultant['绝对高频国家']) else ''
            raw_relative = consultant['相对高频国家'] if pd.notna(consultant['相对高频国家']) else ''
            raw_done = consultant['做过国家'] if pd.notna(consultant['做过国家']) else ''
            
            absolute_high_freq = {country.strip() for country in re.split(r'[、,，\s]+', raw_absolute)} if raw_absolute else set()
            relative_high_freq = {country.strip() for country in re.split(r'[、,，\s]+', raw_relative)} if raw_relative else set()
            done_countries = {country.strip() for country in re.split(r'[、,，\s]+', raw_done)} if raw_done else set()
            
            # 计算各类型匹配的国家
            absolute_matches = case_countries.intersection(absolute_high_freq)
            relative_matches = case_countries.intersection(relative_high_freq)
            done_matches = case_countries.intersection(done_countries)
            
            # 按比例计算分数，考虑加拿大的权重
            if absolute_matches:
                weighted_matches = sum(2 if country == '加拿大' else 1 for country in absolute_matches)
                tag_score_dict['绝对高频国家'] = (tag_weights['绝对高频国家'] / weighted_total) * weighted_matches
            
            # 确保相对高频匹配不与绝对高频匹配重复计算
            relative_matches = relative_matches - absolute_matches
            if relative_matches:
                weighted_matches = sum(2 if country == '加拿大' else 1 for country in relative_matches)
                tag_score_dict['相对高频国家'] = (tag_weights['相对高频国家'] / weighted_total) * weighted_matches
            
            # 确保做过国家匹配不与前两种匹配重复计算
            done_matches = done_matches - absolute_matches - relative_matches
            if done_matches:
                weighted_matches = sum(2 if country == '加拿大' else 1 for country in done_matches)
                tag_score_dict['做过国家'] = (tag_weights['做过国家'] / weighted_total) * weighted_matches
            
        elif case['国家标签'] == '':
            tag_score_dict['绝对高频国家'] = tag_weights['绝对高频国家']
            tag_score_dict['相对高频国家'] = 0
            tag_score_dict['做过国家'] = 0
        
        # 2. 专业标签匹配
        if '专业标签' in case and pd.notna(case['专业标签']):
            case_majors = set(re.split(r'[、,，\s]+', case['专业标签']))
            total_majors = len(case_majors)
            
            absolute_high_freq_majors = set(re.split(r'[、,，\s]+', consultant['绝对高频专业'])) if pd.notna(consultant['绝对高频专业']) else set()
            relative_high_freq_majors = set(re.split(r'[、,，\s]+', consultant['相对高频专业'])) if pd.notna(consultant['相对高频专业']) else set()
            done_majors = set(re.split(r'[、,，\s]+', consultant['做过专业'])) if pd.notna(consultant['做过专业']) else set()
            
            # 计算各类型匹配的专业数量
            absolute_matches = case_majors.intersection(absolute_high_freq_majors)
            relative_matches = case_majors.intersection(relative_high_freq_majors)
            done_matches = case_majors.intersection(done_majors)
            
            # 按比例计算分数
            if absolute_matches:
                tag_score_dict['绝对高频专业'] = (tag_weights['绝对高频专业'] / total_majors) * len(absolute_matches)
            
            # 确保相对高频匹配不与绝对高频匹配重复计算
            relative_matches = relative_matches - absolute_matches
            if relative_matches:
                tag_score_dict['相对高频专业'] = (tag_weights['相对高频专业'] / total_majors) * len(relative_matches)
            
            # 确保做过专业匹配不与前两种匹配重复计算
            done_matches = done_matches - absolute_matches - relative_matches
            if done_matches:
                tag_score_dict['做过专业'] = (tag_weights['做过专业'] / total_majors) * len(done_matches)
        elif case['专业标签'] == '':
            tag_score_dict['绝对高频专业'] = tag_weights['绝对高频专业']
            tag_score_dict['相对高频专业'] = 0
            tag_score_dict['做过专业'] = 0
        
        # 3 博士成功案例和低龄留学成功案例按比例匹配
        proportion_tags = ['博士成功案例', '低龄留学成功案例']
        count = 0
        for tag in proportion_tags:
            if case[tag] == '':
                count += 1
        
        for tag in proportion_tags:
            if pd.notna(case[tag]) and pd.notna(consultant[tag]) and case[tag] != '':
                # 将case和consultant的标签都分割成集合
                case_tags = set(re.split(r'[、,，\s]+', case[tag]))
                consultant_tags = set(re.split(r'[、,，\s]+', consultant[tag]))
                
                # 计算匹配的标签数量
                matched_tags = case_tags.intersection(consultant_tags)
                
                # 如果有匹配的标签，按比例计算得分
                if matched_tags:
                    tag_score_dict[tag] = (tag_weights[tag] / len(case_tags)) * len(matched_tags)

            elif count == 2:
                tag_score_dict[tag] = 5
        # 4. 行业经验标签匹配（反向包含关系：consultant的标签要包含在case中）
        if pd.notna(case['行业经验']) and pd.notna(consultant['行业经验']) and case['行业经验'] != '':
            case_industry = set(re.split(r'[、,，\s]+', case['行业经验']))
            consultant_industry = set(re.split(r'[、,，\s]+', consultant['行业经验']))
            
            # 检查consultant的行业经验是否被case的行业经验包含
            if consultant_industry.issubset(case_industry):
                tag_score_dict['行业经验'] = tag_weights['行业经验']
        elif case['行业经验'] == '':
            tag_score_dict['行业经验'] = tag_weights['行业经验']
        # 5. 直接匹配标签（不需要分割）
        direct_match_tags = [
            '名校专家',
            '文案背景',
            '业务单位所在地'
        ]
        
        for tag in direct_match_tags:
            if case[tag] == '':  # 如果案例标签为空
                tag_score_dict[tag] = tag_weights[tag]
            elif pd.notna(case[tag]) and pd.notna(consultant[tag]):  # 如果案例和顾问标签都不为空
                if case[tag] == consultant[tag]:  # 如果标签匹配
                    tag_score_dict[tag] = tag_weights[tag]
        return  tag_score_dict,direction



    def calculate_workload_score(case, consultant,direction):
        """计算工作量得分"""
        total_score = 0
        if direction == False:
            return total_score
        # 检查学年负荷
        if pd.notna(consultant['学年负荷']):
            value = str(consultant['学年负荷']).lower()
            if value in ['是', 'true', 'yes','有余量']:
                total_score += workload_weights['学年负荷']  # 25分
        
        # 检查近两周负荷
        if pd.notna(consultant['近两周负荷']):
            value = str(consultant['近两周负荷']).lower()
            if value in ['是', 'true', 'yes','有余量']:
                total_score += workload_weights['近两周负荷']  # 25分

        # 检查文书完成率
        if pd.notna(consultant['文书完成率']):
            value = str(consultant['文书完成率']).lower()
            if value in ['是', 'true', 'yes','有余量']:
                total_score += workload_weights['文书完成率']  # 25分

        # 检查申请完成率
        if pd.notna(consultant['申请完成率']):
            value = str(consultant['申请完成率']).lower()
            if value in ['是', 'true', 'yes','有余量']:
                total_score += workload_weights['申请完成率']  # 25分
        
        return total_score

    def calculate_personal_score(case, consultant,direction):
        """计算个人意愿得分"""
        total_score = 0
        if direction == False:
            return total_score
        # 检查个人意愿
        if pd.notna(consultant['个人意愿']):
            value = str(consultant['个人意愿']).lower()
            if value in ['是', 'true', 'yes','接案中']:
                total_score += personal_weights['个人意愿']  # 100分
        
        
        return total_score

    def calculate_final_score(tag_score_dict, consultant, workload_score, personal_score, case):
        """计算最终得分（包含所有维度）"""
        def count_matched_tags(tag_score_dict, case):
            """计算匹配上的标签数量"""
            country_count_need = 0
            special_count_need = 0
            other_count_need = 0
            
            # 国家标签匹配计算
            if any(score > 0 for tag, score in tag_score_dict.items() 
                   if tag in ['绝对高频国家', '相对高频国家','做过国家']):
                # 获取案例中的国家数量
                case_countries = set(re.split(r'[、,，\s]+', case['国家标签'])) if pd.notna(case['国家标签']) else set()
                country_count_need += len(case_countries)
            
            # 特殊标签如果得分，获取案例中对应标签的数量
            special_tags = ['博士成功案例', '低龄留学成功案例','名校专家']
    
            for tag in special_tags:
                if tag_score_dict.get(tag, 0) > 0 and pd.notna(case[tag]) and case[tag] != '':
                    tag_items = set(re.split(r'[、,，\s]+', case[tag]))
                    special_count_need += len(tag_items)

            # 其他标签只要得分就计数
            other_tags = ['绝对高频专业', '相对高频专业', '做过专业','行业经验', '文案背景', 
                          '业务单位所在地']
            for tag in other_tags:
                if tag_score_dict.get(tag, 0) > 0:
                    other_count_need += 1
            
            return country_count_need, special_count_need, other_count_need

        def count_total_consultant_tags(consultant):
            """计算顾问的总标签数"""
            country_count_total = 0
            special_count_total = 0
            other_count_total = 0
            
            # 计算国家标签数
            absolute_high_freq = set(re.split(r'[、,，\s]+', consultant['绝对高频国家'])) if pd.notna(consultant['绝对高频国家']) else set()
            relative_high_freq = set(re.split(r'[、,，\s]+', consultant['相对高频国家'])) if pd.notna(consultant['相对高频国家']) else set()
            done_countries = set(re.split(r'[、,，\s]+', consultant['做过国家'])) if pd.notna(consultant['做过国家']) else set()
            country_count_total += len(absolute_high_freq) + len(relative_high_freq) + len(done_countries)
            
            # 计算特殊标签数
            special_tags = ['博士成功案例', '低龄留学成功案例','名校专家']
    
            for tag in special_tags:
                if pd.notna(consultant[tag]) and consultant[tag] != '':
                    tag_items = set(re.split(r'[、,，\s]+', consultant[tag]))
                    special_count_total += len(tag_items)
            
            # 计算其他标签数
            other_tags = ['绝对高频专业', '相对高频专业','做过专业','行业经验', '文案背景', 
                          '业务单位所在地']
            for tag in other_tags:
                if pd.notna(consultant[tag]) and consultant[tag] != '':
                    other_count_total += 1
                
            return country_count_total, special_count_total, other_count_total
        
        # 计算需求标签数和顾问标签数
        country_count_need, special_count_need, other_count_need = count_matched_tags(tag_score_dict, case)
        country_count_total, special_count_total, other_count_total = count_total_consultant_tags(consultant)
        
        # 计算匹配率
        country_match_ratio = country_count_need / country_count_total if country_count_total > 0 else 1
        special_match_ratio = special_count_need / special_count_total if special_count_total > 0 else 1
        
        # 分别计算国家标签、特殊标签和其他标签的得分
        country_tags_score = sum(score for tag, score in tag_score_dict.items() if tag in ['绝对高频国家', '相对高频国家','做过国家'])
        special_tags_score = sum(score for tag, score in tag_score_dict.items() if tag in [
            '博士成功案例', '低龄留学成功案例','名校专家'
        ])
        
        other_tags_score = sum(score for tag, score in tag_score_dict.items() if tag  in [
            '绝对高频专业', '相对高频专业', '做过专业', '行业经验', '文案背景', '业务单位所在地'
        ])
        # 计算需求标签数
        case_country_count = len(set(re.split(r'[、,，\s]+', case['国家标签']))) if pd.notna(case['国家标签']) else 0
        
        case_special_count = 0
        special_tags = ['博士成功案例', '低龄留学成功案例','名校专家']
        for tag in special_tags:
            if pd.notna(case[tag]) and case[tag] != '':
                case_special_count += len(set(re.split(r'[、,，\s]+', case[tag])))
        
        # 计算需求覆盖率
        country_coverage_ratio = country_count_need / case_country_count if case_country_count > 0 else 1
        special_coverage_ratio = special_count_need / case_special_count if case_special_count > 0 else 1
        
        # 应用匹配率和覆盖率到对应标签得分上
        adjusted_country_score = country_tags_score
        adjusted_special_score = special_tags_score * special_match_ratio * special_coverage_ratio
        
        # 计算调整后的总标签得分
        adjusted_tag_score = adjusted_country_score + adjusted_special_score + other_tags_score
        

        # 计算各维度最终得分
        final_tag_score = (adjusted_tag_score / 100) * dimension_weights['标签匹配'] * 100
        final_workload_score = (workload_score / 100) * dimension_weights['工作量'] * 100
        final_personal_score = (personal_score / 100) * dimension_weights['个人意愿'] * 100
        
        
        # 更新返回结果，添加覆盖率信息
        result = {
            'score': final_tag_score + final_workload_score + final_personal_score,
            'country_count_need': country_count_need,
            'special_count_need': special_count_need,
            'other_count_need': other_count_need,
            'country_count_total': country_count_total,
            'special_count_total': special_count_total,
            'other_count_total': other_count_total,
            'country_match_ratio': country_match_ratio,
            'special_match_ratio': special_match_ratio,
            'country_coverage_ratio': country_coverage_ratio,
            'special_coverage_ratio': special_coverage_ratio,
            'country_tags_score': country_tags_score,
            'special_tags_score': special_tags_score,
            'other_tags_score': other_tags_score
        }
        
        return result

    def find_best_matches(consultant_tags_file, merge_df, area):
        """找到每条案例得分最高的顾问们"""
        # 存储所有案例的匹配结果
        all_matches = {}
        all_tag_score_dicts = {}  # 存储每个案例的所有顾问得分字典
        all_workload_score_dicts = {}  # 存储每个案例的所有顾问工作量得分字典
        all_completion_rate_score_dicts = {}  # 存储每个案例的所有顾问完成率得分字典
        all_local_consultants = {}
        # 对每条案例进行匹配
        for idx, case in merge_df.iterrows():
            scores = []
            case_tag_score_dicts = {}  # 存储当前案例的所有顾问得分字典
            case_workload_score_dicts = {}  # 存储当前案例的所有顾问工作量得分字典
            
            
            # 选择顾问范围（本地或全部）
            local_consultants = consultant_tags_file[consultant_tags_file['文案顾问业务单位'] == case['文案顾问业务单位']]
            all_consultants = consultant_tags_file
            consultants = local_consultants if area else all_consultants
            
            # 计算每个顾问对当前案例的得分
            for cidx, consultant in consultants.iterrows():
                try:
                    direction = True
                    # 获取标签匹配得分和得分字典
                    tag_score_dict,direction = calculate_tag_matching_score(case, consultant,direction)
                    
                    workload_score = calculate_workload_score(case, consultant,direction)
                    
                    personal_score = calculate_personal_score(case, consultant,direction)
                    
                    # 计算最终得分
                    final_result = calculate_final_score(
                        tag_score_dict,
                        consultant,
                        workload_score,
                        personal_score,
                        case
                    )
                    
                    # 保存当前顾问的得分字典
                    case_tag_score_dicts[consultant['文案顾问']] = tag_score_dict
                    case_workload_score_dicts[consultant['文案顾问']] = workload_score
                    
                    
                    # 获取顾问的标签数据
                    consultant_info = {
                        'name': consultant['文案顾问'],
                        'businessunits': consultant['文案顾问业务单位'],
                        'area':area,
                        'score': final_result['score'],
                        'tag_score_dict': tag_score_dict,
                        'workload_score': workload_score,
                        'personal_score': personal_score,
                        'country_count_need': final_result['country_count_need'],
                        'special_count_need': final_result['special_count_need'],
                        'other_count_need': final_result['other_count_need'],
                        'country_count_total': final_result['country_count_total'],
                        'special_count_total': final_result['special_count_total'],
                        'other_count_total': final_result['other_count_total'],
                        'country_match_ratio': final_result['country_match_ratio'],
                        'special_match_ratio': final_result['special_match_ratio'],
                        'country_coverage_ratio': final_result['country_coverage_ratio'],
                        'special_coverage_ratio': final_result['special_coverage_ratio'],
                        'country_tags_score': final_result['country_tags_score'],
                        'special_tags_score': final_result['special_tags_score'],
                        'other_tags_score': final_result['other_tags_score']
                    }
                    
                    # 安全地添加标签字段
                    standard_fields = [
                        '绝对高频国家', '相对高频国家', '做过国家','绝对高频专业', '相对高频专业', '做过专业','文案背景',
                        '行业经验', '业务单位所在地', '学年负荷', '近两周负荷', '文书完成率', '申请完成率', '个人意愿',
                        '名校专家', '博士成功案例', '低龄留学成功案例','文案方向'
                    ]
                    
                    for field in standard_fields:
                        if field in consultant.index and pd.notna(consultant[field]):
                            consultant_info[field] = consultant[field]
                        else:
                            consultant_info[field] = ''
                    
                    scores.append(consultant_info)
                except Exception as e:
                    st.error(f"计算得分出错: {str(e)}")
            
            # 按得分降序排序
            scores.sort(key=lambda x: -x['score'])
            
            # 选择得分最高的顾问们
            selected_consultants = []
            
            # 首先筛选出符合国家标签条件的顾问
            qualified_scores = [
                s for s in scores 
                if (s['tag_score_dict'].get('绝对高频国家', 0) + 
                    s['tag_score_dict'].get('相对高频国家', 0)) > 0
            ]
            
            # 不符合国家标签条件的顾问
            unqualified_scores = [
                s for s in scores 
                if (s['tag_score_dict'].get('绝对高频国家', 0) + 
                    s['tag_score_dict'].get('相对高频国家', 0)) == 0
            ]
            
            def create_consultant_data(s):
                """创建顾问数据结构"""
                consultant_data = {
                    'display': f"{s['name']}（{s['score']:.1f}分）",
                    'name': s['name'],
                    'businessunits': s['businessunits'],
                    'area': s['area'],
                    'score': s['score'],
                    'tag_score_dict': s['tag_score_dict'],
                    'workload_score': s['workload_score'],
                    'personal_score': s['personal_score'],
                    'country_count_need': s['country_count_need'],
                    'special_count_need': s['special_count_need'],
                    'other_count_need': s['other_count_need'],
                    'country_count_total': s['country_count_total'],
                    'special_count_total': s['special_count_total'],
                    'other_count_total': s['other_count_total'],
                    'country_match_ratio': s['country_match_ratio'],
                    'special_match_ratio': s['special_match_ratio'],
                    'country_coverage_ratio': s['country_coverage_ratio'],
                    'special_coverage_ratio': s['special_coverage_ratio'],
                    'country_tags_score': s['country_tags_score'],
                    'special_tags_score': s['special_tags_score'],
                    'other_tags_score': s['other_tags_score']
                }
                
                # 复制其他标签字段
                for field in standard_fields:
                    if field in s:
                        consultant_data[field] = s[field]
                
                return consultant_data
            
            # 如果符合条件的顾问超过9个
            if len(qualified_scores) >= 9:
                # 获取第9高分
                ninth_score = qualified_scores[8]['score']
                # 选择所有大于等于第9高分的顾问（处理同分情况）
                selected_consultants = [create_consultant_data(s) for s in qualified_scores if s['score'] >= ninth_score]
            else:
                # 如果符合条件的顾问不足9个
                # 先添加所有符合条件的顾问
                selected_consultants = [create_consultant_data(s) for s in qualified_scores]
                
                # 计算还需要补充的顾问数量
                remaining_slots = 9 - len(selected_consultants)
                
                if remaining_slots > 0 and unqualified_scores:
                    # 从不符合条件的顾问中取得分最高的若干个
                    # 获取第remaining_slots位的分数
                    cutoff_score = unqualified_scores[remaining_slots - 1]['score'] if len(unqualified_scores) >= remaining_slots else unqualified_scores[-1]['score']
                    # 添加所有大于等于这个分数的不符合条件顾问（处理同分情况）
                    additional_consultants = [create_consultant_data(s) for s in unqualified_scores if s['score'] >= cutoff_score]
                    selected_consultants.extend(additional_consultants)
            
            # 存储当前案例的匹配结果和所有顾问的得分字典
            case_key = f"案例{idx + 1}"
            all_matches[case_key] = selected_consultants
            all_tag_score_dicts[case_key] = case_tag_score_dicts
            all_workload_score_dicts[case_key] = case_workload_score_dicts
            
        

        return all_matches, all_tag_score_dicts, all_workload_score_dicts, 
    
    # 1. 先计算本地顾问的得分
    area = True
    local_scores, all_tag_score_dicts, all_workload_score_dicts = find_best_matches(consultant_tags_file, merge_df, area)

    # 2. 检查7个判断条件
    def all_conditions_met(all_tag_score_dicts, all_workload_score_dicts, case,idx):
        # 获取所有顾问列表
        consultants = set(all_tag_score_dicts["案例1"].keys())
        
        idx_case = f"案例{idx+1}"
        # 对每个顾问单独判断所有条件
        for consultant in consultants:
            # 初始化该顾问的标志
            consultant_conditions = {
                '国家标签': False,
                '博士成功案例': False,
                '低龄留学成功案例': False,
                '行业经验': False,
                '工作量': False,
                
            }
            
            # 1. 国家标签判断
            has_country = True if case['国家标签'] != '' else False
            case_country_count = len(set(re.split(r'[、,，\s]+', case['国家标签']))) if pd.notna(case['国家标签']) else 0

            if has_country :
                tag_score_dicts = all_tag_score_dicts[idx_case]
                tag_score_dict = tag_score_dicts[consultant]
                score_country_high = 0
                score_country_done = 0
                for tag, score in tag_score_dict.items():
                    if tag in ['绝对高频国家', '相对高频国家']:
                        score_country_high += score
                    if tag in ['做过国家']:
                        score_country_done += score
                if score_country_high <= tag_weights['绝对高频国家'] and score_country_high >= tag_weights['相对高频国家'] and score_country_done == 0:
                    consultant_conditions['国家标签'] = True
            else:
                consultant_conditions['国家标签'] = True
                
            
            # 2. 名校专家标签判断
            has_school = True if case['名校专家'] != '' else False
            if has_school:
                tag_score_dicts = all_tag_score_dicts[idx_case]
                tag_score_dict = tag_score_dicts[consultant]
                if '名校专家' in tag_score_dict and tag_score_dict['名校专家'] == tag_weights['名校专家']:
                    consultant_conditions['名校专家'] = True
            else:
                consultant_conditions['名校专家'] = True
            # 3. 博士成功案例标签判断
            has_doctor = True if case['博士成功案例'] != '' else False
            if has_doctor:
                tag_score_dicts = all_tag_score_dicts[idx_case]
                tag_score_dict = tag_score_dicts[consultant]
                if '博士成功案例' in tag_score_dict and tag_score_dict['博士成功案例'] == tag_weights['博士成功案例']:
                    consultant_conditions['博士成功案例'] = True
            else:
                consultant_conditions['博士成功案例'] = True
            # 4. 低龄留学成功案例标签判断
            has_lowage = True if case['低龄留学成功案例'] != '' else False
            if has_lowage:
                tag_score_dicts = all_tag_score_dicts[idx_case]
                tag_score_dict = tag_score_dicts[consultant]
                if '低龄留学成功案例' in tag_score_dict and tag_score_dict['低龄留学成功案例'] == tag_weights['低龄留学成功案例']:
                    consultant_conditions['低龄留学成功案例'] = True
            else:
                consultant_conditions['低龄留学成功案例'] = True
            # 5. 行业经验标签判断
            has_industry = True if  str(case['行业经验']) == '专家' else False
            if has_industry:
                tag_score_dicts = all_tag_score_dicts[idx_case]
                tag_score_dict = tag_score_dicts[consultant]
                if '行业经验' in tag_score_dict and tag_score_dict['行业经验'] > 0:
                    consultant_conditions['行业经验'] = True
            else:
                consultant_conditions['行业经验'] = True
            # 6. 工作量标签判断
            
            workload_score_dicts = all_workload_score_dicts[idx_case]
            
            workload_score_dict = workload_score_dicts[consultant]
            if workload_score_dict == 100:
                consultant_conditions['工作量'] = True
            # 如果这个顾问满足所有条件，直接返回True
            if all(consultant_conditions.values()):
                
                return True

                
                
        # 如果没有任何顾问满足所有条件，返回False
        return False
    for idx,case in merge_df.iterrows():
        if all_conditions_met(all_tag_score_dicts, all_workload_score_dicts, case,idx):
            # 如果所有条件都满足，使用本地顾问的匹配结果
            return local_scores ,area
    else:
        # 如果有任何条件不满足，使用所有顾问重新计算匹配

        area = False
        all_scores,all_tag_score_dicts,all_workload_score_dicts = find_best_matches(consultant_tags_file, merge_df, area)
        return all_scores ,area


