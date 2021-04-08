
import os 
import csv
import pandas as pd
import numpy as np
import streamlit as st
import torch
from transformers import TapasTokenizer, TapasForQuestionAnswering

def load_model():
    print('downloading model')
    model_name = 'google/tapas-base-finetuned-wtq'
    model = TapasForQuestionAnswering.from_pretrained(model_name)
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    print('model downloaded')
    return model,tokenizer

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Query your Table using TAPAS')

uploaded_file = st.file_uploader("Choose your CSV file",type = 'csv')

placeholder = st.empty()

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.replace(',','', regex=True, inplace=True)
    if st.checkbox('Want to see the data?'):
        placeholder.dataframe(data)

st.header('Enter your queries')

input_queries = st.text_input('Type your queries separated by comma(,)',value='')
input_queries = input_queries.split(',')

colors1 = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(input_queries))]
colors2 = ['background-color:'+str(color)+'; color: black' for color in colors1]

def styling_specific_cell(x,tags,colors):
    df_styler = pd.DataFrame('', index=x.index, columns=x.columns)
    for idx,tag in enumerate(tags):
        for r,c in tag:
            df_styler.iloc[r, c] = colors[idx]
    return df_styler
    
if st.button('Predict Answers'):
    with st.spinner('It will take approx a minute'):
        model,tokenizer = load_model()
        print('fetching predictions')
        data = data.astype(str)
        inputs = tokenizer(table=data, queries=input_queries, padding='max_length', return_tensors="pt")
        outputs = model(**inputs)
        predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions( inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())
        print('prediction done')
        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3:"COUNT"}
        aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]
    
        answers = []
        
        for coordinates in predicted_answer_coordinates:
           if len(coordinates) == 1:
             # only a single cell:
             answers.append(data.iat[coordinates[0]])
           else:
             # multiple cells
             cell_values = []
             for coordinate in coordinates:
                cell_values.append(data.iat[coordinate])
             answers.append(", ".join(cell_values))
             
    st.success('Done! Please check below the answers and its cells highlighted in table above')
    
    placeholder.dataframe(data.style.apply(styling_specific_cell,tags=predicted_answer_coordinates,colors=colors2,axis=None))
      
    for query, answer, predicted_agg, c in zip(input_queries, answers, aggregation_predictions_string, colors1):
        st.write('\n')
        st.markdown('<font color={} size=4>**{}**</font>'.format(c,query), unsafe_allow_html=True)
        st.write('\n')
        if predicted_agg == "NONE" or predicted_agg == 'COUNT':
            st.markdown('**>** '+str(answer))
        else:
            answer = np.array(answer.split(','))
            answer = [i.strip() for i in answer]
            answer = [float(i) for i in answer]
            if predicted_agg == 'SUM':
                st.markdown('**>** '+str(np.sum(answer)))
            else:
                st.markdown('**>** '+str(np.round(np.mean(answer),2)))
