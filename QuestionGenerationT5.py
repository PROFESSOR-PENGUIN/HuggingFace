from transformers import AutoTokenizer
from fastT5 import OnnxT5,get_onnx_runtime_sessions
from fastT5 import generate_onnx_representation,quantize
import os

default_context = """Another important distinction is between companies that build enterprise products (B2B - business to business) and companies that build customer products (B2C - business to consumer).
B2B companies build products for organizations. Examples of enterprise products are Customer relationship management (CRM) software, project management tools, database management systems, cloud hosting services, etc.
B2C companies build products for individuals. Examples of consumer products are social networks, search engines, ride-sharing services, health trackers, etc.
Many companies do both -- their products can be used by individuals but they also offer plans for enterprise users. For example, Google Drive can be used by anyone but they also have Google Drive for Enterprise.
Even if a B2C company doesn’t create products for enterprises directly, they might still need to sell to enterprises. For example, Facebook’s main product is used by individuals but they sell ads to enterprises. Some might argue that this makes Facebook users products, as famously quipped: “If you’re not paying for it, you’re not the customer; you’re the product being sold.”"""

default_answer = "consumer products"


t5_model_path="onnx_t5"
t5_chkpt="mrm8488/t5-base-finetuned-question-generation-ap"

#function to generate ONNX model for any given chkpt on HuggingFace
def createT5OnnxModel(chkpt=t5_chkpt,
                      model_path=t5_model_path):
    """

    :param chkpt:
    :param model_path:
    :return:
    """
    # Step 1. convert huggingfaces t5 model to onnx
    onnx_model_paths = generate_onnx_representation(chkpt,output_path=model_path)

    # Step 2. quantize the converted model for fast inference and to reduce model size.
    quant_model_paths = quantize(model_path)

    # delete non-quantized models to save space
    try:
        os.remove(f'{model_path}/{chkpt.split("/")[1]}-encoder.onnx')
        os.remove(f'{model_path}/{chkpt.split("/")[1]}-decoder.onnx')
        os.remove(f'{model_path}/{chkpt.split("/")[1]}-init-decoder.onnx')
    except:
        pass

model_path_quanitzed=(f'{t5_model_path}/{t5_chkpt.split("/")[1]}-encoder-quantized.onnx',
                      f'{t5_model_path}/{t5_chkpt.split("/")[1]}-decoder-quantized.onnx',
                      f'{t5_model_path}/{t5_chkpt.split("/")[1]}-init-decoder-quantized.onnx'
                      )
tokenize=AutoTokenizer.from_pretrained(t5_chkpt)

def create_question_t5(model,tokenizer,context,answer,max_length=64):
    input="context: %s answer: %s </s>" % (context,answer)
    features=tokenizer([input],return_tensors='pt')
    output=model.generate(input_ids=features['input_ids'],
                          attention_mask=features['attention_mask'],
                          max_length=max_length,
                          num_beams=3)
    return tokenizer.decode(output.squeeze(),skip_special_tokens=True)

def create_answers_t5(model,tokenizer,context,question,max_length=128):
    input = "context: %s question: %s </s>" % (context, question)
    features=tokenizer([input],return_tensors='pt')
    output=model.generate(input_ids=features['input_ids'],
                          attention_mask=features['attention_mask'],
                          max_length=max_length,
                          num_beams=3)

    return tokenizer.decode(output.squeeze(), skip_special_tokens=True)


#load the models and create session
model_session=get_onnx_runtime_sessions(model_paths=model_path_quanitzed,n_threads=1,parallel_exe_mode=True)
model_t5=OnnxT5(model_or_model_path=t5_chkpt,onnx_model_sessions=model_session)
