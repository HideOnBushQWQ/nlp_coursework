"""
实体展示模块
"""

from typing import List, Dict, Any
from IPython.display import HTML
import html


class EntityDisplayer:
    """实体可视化展示器"""

    def __init__(self):
        """初始化展示器"""
        self.colors = {
            'PER': '#FFB6C1',  # 粉色 - 人名
            'LOC': '#87CEEB',  # 天蓝色 - 地名
            'ORG': '#98FB98',  # 淡绿色 - 机构名
            'MISC': '#FFD700'  # 金色 - 其他
        }

    def display_entities(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        show_label: bool = True
    ) -> str:
        """
        生成实体高亮的HTML

        Args:
            text: 原始文本
            entities: 实体列表
            show_label: 是否显示标签

        Returns:
            HTML字符串
        """
        if not entities:
            return html.escape(text)

        # 将文本拆分为字符列表（处理中英文混合）
        chars = list(text)

        # 创建字符到实体的映射
        char_to_entity = {}
        for entity in entities:
            entity_text = entity['text']
            entity_type = entity['type']

            # 在原文本中查找实体位置
            start_pos = text.find(entity_text)
            if start_pos != -1:
                for i in range(start_pos, start_pos + len(entity_text)):
                    char_to_entity[i] = entity_type

        # 生成HTML
        html_parts = []
        i = 0
        while i < len(chars):
            if i in char_to_entity:
                # 找到实体的起始和结束
                entity_type = char_to_entity[i]
                start = i
                while i < len(chars) and char_to_entity.get(i) == entity_type:
                    i += 1
                end = i

                # 提取实体文本
                entity_text = ''.join(chars[start:end])

                # 生成高亮HTML
                color = self.colors.get(entity_type, '#CCCCCC')
                entity_html = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 0 1px;">'
                entity_html += html.escape(entity_text)
                if show_label:
                    entity_html += f'<sub style="font-size: 0.7em; color: #666; font-weight: bold;"> {entity_type}</sub>'
                entity_html += '</span>'
                html_parts.append(entity_html)
            else:
                html_parts.append(html.escape(chars[i]))
                i += 1

        return ''.join(html_parts)

    def display_entities_list(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> str:
        """
        以列表形式展示实体

        Args:
            text: 原始文本
            entities: 实体列表

        Returns:
            HTML字符串
        """
        if not entities:
            return "<p>No entities found.</p>"

        html_parts = ['<div style="font-family: Arial, sans-serif;">']
        html_parts.append(f'<p><strong>Text:</strong> {html.escape(text)}</p>')
        html_parts.append('<p><strong>Entities:</strong></p>')
        html_parts.append('<ul style="list-style-type: none; padding-left: 0;">')

        for entity in entities:
            color = self.colors.get(entity['type'], '#CCCCCC')
            html_parts.append(
                f'<li style="margin: 5px 0;">'
                f'<span style="background-color: {color}; padding: 3px 8px; border-radius: 3px; display: inline-block; min-width: 80px; text-align: center;">'
                f'{entity["type"]}'
                f'</span> '
                f'<strong>{html.escape(entity["text"])}</strong> '
                f'<span style="color: #666;">({entity["start"]}, {entity["end"]})</span>'
                f'</li>'
            )

        html_parts.append('</ul>')
        html_parts.append('</div>')

        return ''.join(html_parts)

    def display_jupyter(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        mode: str = 'inline'
    ):
        """
        在Jupyter Notebook中显示

        Args:
            text: 原始文本
            entities: 实体列表
            mode: 显示模式，'inline'或'list'

        Returns:
            IPython HTML对象
        """
        if mode == 'inline':
            html_str = self.display_entities(text, entities)
        else:  # mode == 'list'
            html_str = self.display_entities_list(text, entities)

        return HTML(html_str)

    def display_comparison(
        self,
        text: str,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> str:
        """
        对比显示预测结果和真实标签

        Args:
            text: 原始文本
            predictions: 预测的实体
            ground_truth: 真实的实体

        Returns:
            HTML字符串
        """
        html_parts = ['<div style="font-family: Arial, sans-serif;">']

        html_parts.append('<h3>Prediction:</h3>')
        html_parts.append('<div style="padding: 10px; background-color: #f9f9f9; border-radius: 5px; margin: 10px 0;">')
        html_parts.append(self.display_entities(text, predictions))
        html_parts.append('</div>')

        html_parts.append('<h3>Ground Truth:</h3>')
        html_parts.append('<div style="padding: 10px; background-color: #f9f9f9; border-radius: 5px; margin: 10px 0;">')
        html_parts.append(self.display_entities(text, ground_truth))
        html_parts.append('</div>')

        # 计算匹配情况
        pred_set = set((e['text'], e['type']) for e in predictions)
        true_set = set((e['text'], e['type']) for e in ground_truth)

        correct = len(pred_set & true_set)
        precision = correct / len(pred_set) if pred_set else 0
        recall = correct / len(true_set) if true_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        html_parts.append('<h3>Metrics:</h3>')
        html_parts.append('<ul>')
        html_parts.append(f'<li>Precision: {precision:.2%}</li>')
        html_parts.append(f'<li>Recall: {recall:.2%}</li>')
        html_parts.append(f'<li>F1-Score: {f1:.2%}</li>')
        html_parts.append('</ul>')

        html_parts.append('</div>')

        return ''.join(html_parts)

    def save_html(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        save_path: str,
        mode: str = 'inline'
    ):
        """
        保存为HTML文件

        Args:
            text: 原始文本
            entities: 实体列表
            save_path: 保存路径
            mode: 显示模式
        """
        if mode == 'inline':
            content = self.display_entities(text, entities)
        else:
            content = self.display_entities_list(text, entities)

        html_template = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NER Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 50px auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .content {{
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>Named Entity Recognition Results</h1>
    <div class="content">
        {content}
    </div>
</body>
</html>
'''

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_template)

        print(f"HTML saved to {save_path}")
