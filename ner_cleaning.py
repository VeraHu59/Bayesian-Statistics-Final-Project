import pandas as pd
import numpy as np
def read_data(data_path):
    words = []
    pos_list = []
    with open(data_path,'rb') as f:
        for line in f:
            line = line.decode(errors='ignore')
            # print(line)
            if len(line.strip().split("\t"))==2:
                word, pos = line.strip().split("\t") 
            # pre_pos = line.split(",")[1]
            # pre_pre_pos = line.split(",")[2]
            # sentence_id = line.split(",")[3]
            # word = line.split(",")[0]
            # if str.isdigit(word):
            #     print(word, pos)
                words.append(word)
                pos_list.append(pos)
    return words, pos_list
    
def write_data(word, pos, output_path):
    with open(output_path,'w') as f:
        for i in range(len(word)):
            f.write(word[i]+'\t'+pos[i]+"\n")
    print(len(word))
    print(len(set(pos)))
    print(set(pos))
    f.close()

def initial_transition(input_path):
    count = {}
    total = 0
    with open(input_path,'r') as f:
        data = f.read().strip().split('\n')
        for line in data:
            total += 1
            # print(line)
            word, p = line.strip().split("\t")
            if p in count:
                count[p] = count[p]+1
            else:
                count[p] = 1
    # print(count)
    new_count = {}
    for key,value in count.items():
        new_count[key] = value/total
    # print(new_count)
    with open('./NER/ner_initial_transition.csv','w') as f:
        for key, value in new_count.items():
            f.write(key+","+str(value)+"\n")
    f.close()
    
def emit_prob(path):
    emit = {}
    with open(path,'r') as f:
        data = f.read().strip().split('\n')
        for line in data:
            word,pos = line.strip().split("\t")
            if pos in emit:
                if word in emit[pos]:
                    emit[pos][word] = emit[pos][word] + 1
                else:
                    emit[pos][word]=1
            else:
                emit[pos] = {}
                emit[pos][word]=1
 
    df = pd.DataFrame(emit).fillna(0)
    col = df.columns
    s = pd.Series(df.sum(axis=0))
    for i in range(len(col)):
        df[col[i]] = df[col[i]].apply(lambda x: x/s[i])

    return df

                
def transition(path):
    pos_list = []
    with open(path, 'r') as f:
        data = f.read().strip().split('\n')
        for line in data:
            word, pos = line.strip().split('\t')
            pos_list.append(pos)
    transition_matrix = {}
    for i in range(1,len(pos_list)):
        if pos_list[i] in transition_matrix:
            if pos_list[i-1] in transition_matrix[pos_list[i]]:
                transition_matrix[pos_list[i]][pos_list[i-1]] = transition_matrix[pos_list[i]][pos_list[i-1]]+1
            else:
                transition_matrix[pos_list[i]][pos_list[i-1]] = 1
        else:
                transition_matrix[pos_list[i]] = {}
                transition_matrix[pos_list[i]][pos_list[i-1]] = 1
    df = pd.DataFrame(transition_matrix).fillna(0)
    col = df.columns
    df['sum'] = df.sum(axis=1)
    m = df.shape[0]
    df = [df.iloc[:,i]/df['sum'] for i in range(m)]
    df = pd.concat(df, axis=1)
    print(df)
    # df = df.drop(columns=['sum'],axis=1)
    # print(col)
    # print(df)
    df.columns = col
    # print(df)
    return df    
if __name__=="__main__":
    word, pos = read_data("/Users/husiyun/Desktop/927 Project/NER/ner.txt")
    write_data(word,pos,'./NER/cleaned_ner.txt')
    initial_transition('/Users/husiyun/Desktop/927 Project/NER/cleaned_ner.txt')
    df = emit_prob("/Users/husiyun/Desktop/927 Project/NER/cleaned_ner.txt")
    df.to_csv('./NER/ner_emit.csv')
    df2 = transition("/Users/husiyun/Desktop/927 Project/NER/cleaned_ner.txt")
    df2.to_csv('./NER/ner_transition.csv')
    