import re


def replace_patterns(text):
    # 正则表达式匹配IP地址
    ip_pattern = r'\b\d{1,3}(?:\[.\]\d{1,3}){3}\b'
    text = re.sub(ip_pattern, "[IP]", text)

    # 正则表达式匹配简单域名
    domain_pattern = r'\b[\w-]+(?:\[.\][\w-]+)+\b'
    text = re.sub(domain_pattern, "[DOMAIN]", text)

    # 更新的正则表达式匹配复杂URL
    url_pattern = r'http[S]?://\S+'
    text = re.sub(url_pattern, "[URL]", text)

    return text


# 示例文本
sample_text = "Check the following IP 192[.]168[.]1[.]1 and visit cdn[.]ns[.]time12[.]cf or http://103.224.80[.]44:8080/kernel for more details."

# 调用函数
replaced_text = replace_patterns(sample_text)
print(replaced_text)
