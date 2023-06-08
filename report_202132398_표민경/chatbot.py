import pandas as pd
import numpy as np

import os, sys
class SimpleChatBot:
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)

    def load_data(self, filepath):
        data = pd.read_csv(os.path.join(sys.path[0], filepath))
        questions = data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
        answers = data['A'].tolist()   # 답변열만 뽑아 파이썬 리스트로 저장
        return questions, answers

    def revenshtein_distance(self, a, b): # 문장 vs 문장의 레벤슈타인 거리 계산 함수
        a_len = len(a) # a 길이
        b_len = len(b) # b 길이

        if a == b : # 더해야 하는 길이, 문장이 같으면 dist = 0
            return 0           
        # 둘중 하나의 길이가 0이면, dist는 다른 하나의 길이
        elif a_len == 0 : 
            return b_len
        elif b_len == 0 :
            return a_len

        # 2차원 matrix 생성
        # Matrix 초기화 예시
        # [0, 1, 2, 3]
        # [1, 0, 0, 0]
        # [2, 0, 0, 0]
        # [3, 0, 0, 0] 

        matrix = [[] for _ in range(a_len+1)] #a_len+1개의 행 생성
        for i in range(a_len+1) :
            matrix[i] = [0 for _ in range(b_len+1)] #b_len+1 개의 열 생성
        # 인덱스, 컬럼 초기값 설정
        # 행인덱스 설정
        for i in range(a_len+1):
            matrix[i][0] = i
        # 열인덱스 설정
        for j in range(b_len+1):
            matrix[0][j] = j
        
        # 레벤슈타인 거리 계산하기 ( a : 행, b : 열 )
        for i in range(1, a_len+1): # 행열인덱스 제외 매트릭스 내부 계산을 위해 1부터
            ac = a[i-1] # a문장의 i번째 character
            for j in range(1, b_len+1) : #행열인덱스제외 매트릭스 내부 계산을 위해 1부터
                bc = b[j-1] # b문장의 j번째 character

                # 거리 계산 rule
                add_rule = matrix[i][j-1] + 1             # 문자 삽입 시, 왼쪽 수 +1
                del_rule = matrix[i-1][j] + 1             # 문자 제거 시, 위쪽 수 +1
                edit_cost = 0 if (ac == bc) else 1
                edit_rule = matrix[i-1][j-1] + edit_cost  # 문자 변경 시, 대각선 +1, 동일 문자는 대각선 그대로

                # 거리 계산 rule 중 가장 낮은 수 입력
                matrix[i][j] = min([add_rule, del_rule, edit_rule])

        # 최종 레벤슈타인 거리 : 마지막 행,열 값
        return matrix[a_len][b_len]
    
    def find_best_answer(self, input_sentence):

        rev_distance_list = [] # 입력 문장과 전체 질문간의 레벤슈타인 거리를 저장할 리스트
        # 전체 문장에서 개별 문장을 뽑아서 레벤슈타인 거리 계산  
        for question_i in self.questions :    
            rev_distance = self.revenshtein_distance(input_sentence, question_i) # 레벤슈타인 거리 계산
            rev_distance_list.append(rev_distance) # 각 질문 문장과 입력 문장에 대한 레벤슈타인 계산 거리 list에 추가
        
        # 레벨슈타인 거리가 가장 작은 값의 인덱스가 유사도 높음 argmin 사용
        best_match_index = np.array(rev_distance_list).argmin()
        return self.answers[best_match_index] # 레벤슈타인 거리가 가장 작은 질문에 해당되는 대답 (유사 질문의 대답)


# CSV 파일 경로를 지정하세요.
filepath = 'ChatbotData.csv'

# 간단한 챗봇 인스턴스를 생성합니다.
chatbot = SimpleChatBot(filepath)

# '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복합니다.
while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    response = chatbot.find_best_answer(input_sentence)
    print('Chatbot:', response)
    
