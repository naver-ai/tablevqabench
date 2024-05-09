"""
TableVQA-Bench
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""

VWTQ_PROMPT = (
        "You are asked to answer questions asked on an image.\n"   
        "You should answer the question with a single word.\n"
        "Example: \n"
        "Question: what was the only year mr. wu competed in the olympic games?\n"
        "Answer: 2004\n"
        "Question: which township in pope county, arkansas has the least amount of water area?\n"
        "Answer: Freeman\n"
        "If you have multiple answers, please separate them with || marks. Example: Apple||Banana||Tomato\n\n"                     
        "Question: {question}\n"                                  
        "Answer:"
)

VTABFACT_PROMPT = (
            "You are asked to answer whether the statement is True or False based on given image\n"   
            "You should only answer true or false.\n"       
            "Example: \n"
            "Statement: the milwaukee buck win 6 game in the 2010 - 11 season\n"
            "Answer: True\n"
            "Statement: only the top team score above the average of 8.8\n"
            "Answer: False\n\n"            
            "Statement: {question}\n"
            "Answer:"                   
        )

FINTABNETQA_PROMPT = (
        "You are asked to answer questions asked on a image.\n"   
        "You should answer the question within a single word or few words.\n"
        "If units can be known, the answer should include units such as $, %, million and etc.\n"
        "Example: \n"
        "Question: What were the total financing originations for the fiscal year ended October 31, 2004?\n"
        "Answer: $3,852 million\n"
        "Question: What is the time period represented in the table?\n"
        "Answer: October 31\n"
        "Question: What was the percentage of net sales for selling, general and administrative expenses in 2006?\n"
        "Answer: 34.2%\n"                     
        "Question: {question}\n"                                  
        "Answer:"
)