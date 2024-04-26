from flask import Flask, request, jsonify, render_template, url_for,session
import base64
from io import BytesIO
import pandas as pd
import os
import matplotlib.pyplot as plt
from pca import *
from dash import Dash, html, dcc


POISONED_WORKER_IDS = []
file_path = None

app = Flask(__name__)
app.secret_key = 'your_very_secret_key_here'


def all_f1(option1_v, option2_v, optionCI_v, slider5, slider4, slider3, range_val):
    if option1_v == 'Label Flipping Attack':
        pworker = slider5
        dataset = optionCI_v
        name = 'la'
        fname= name + '_' +dataset + '_' + pworker
        filename = name + '_' +dataset + '_' + pworker + '_3000.log'
    elif option1_v == "Attack Timing":
        pworker = slider5
        timing = slider4
        name = 'at'
        print(option2_v)
        if option2_v == 'Attack Entry Point':
            pre = 'post'
        else:
            pre = 'pre'
        fname= name + '_' + pre + '_' + pworker + '_' + timing
        filename = name + '_' + pre + '_' + pworker + '_' + timing + '_3000.log'
    else:
        name = 'ma'
        pworker = slider5
        timing = slider4
        if option2_v == 'Attack Entry Point':
            pre = 'post'
        else:
            pre = 'pre'
        probability = slider3
        fname = name + '_' + pre + '_' + pworker + '_' + timing + '_' + probability
        filename = name + '_' + pre + '_' + pworker + '_' + timing + '_' + probability +'_3000.log'

    directory_path = 'data/f1/'
    global file_name
    file_name = fname
    global file_path
    file_path= os.path.join(directory_path, filename)

    tags = []
    f1_scores = []
    supports = []

    # Open the log file and read its contents
    with open(file_path, "r") as file:
        for line in file:
            # Check if line starts with "accuracy"
            if line.strip().startswith("accuracy"):
                # Split the line by whitespace
                parts = line.split()
                tags.append(parts[0])
                f1_scores.append(float(parts[1]))
                supports.append(int(parts[2]))

    # Create a dataframe from the extracted values
    df = pd.DataFrame({
        "tag": tags,
        "f1": f1_scores,
        "support": supports
    })

    # Using the 'bmh' style context for this plot
    with plt.style.context('bmh'):
        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(28, 14))

        # Plotting the F1 score
        ax.plot(df.index, df['f1'], label='F1 Score')  # Assuming 'f1' is the column name

        # Highlight the range from index 10 to 25
        if option1_v != 'Label Flipping Attack':
            a = int(slider4)- 10
            b = int(slider4)+ 10
            ax.axvspan(a, b, color='red', alpha=0.3)

        # Adding labels and title
        ax.set_xlabel('Round')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score by Round')
        ax.set_ylim(0, 1)

        # Add a legend
        ax.legend()

        fig.savefig('./static/css/overall.jpg')

        #get Poisoning
        with open(file_path, "r") as file:
            for line in file:
                # Check if line starts with "accuracy"
                if "Poisoning data for workers:" in line:
                    # Find the start of the list in the line
                    start_index = line.index('[')
                    # Find the end of the list in the line
                    end_index = line.index(']')
                    # Extract the list portion and remove whitespace
                    list_str = line[start_index + 1:end_index].strip()
                    # Convert the string numbers into integers
                    p_list = [int(num.strip()) for num in list_str.split(',')]
                    break  # Assuming we only need the first occurrence

        global POISONED_WORKER_IDS
        POISONED_WORKER_IDS = p_list
        #get avaliable workers
        all_numbers = set(range(50))
        # Convert p_list to a set
        poison_set = set(p_list)
        # Subtract the set of poisoned workers from all numbers
        available_numbers = list(all_numbers - poison_set)
        # Return the list of available numbers
        return p_list, available_numbers



def target_f1(file_path,option1_v, option2_v, optionCI_v, slider5, slider4, slider3, range_val):
    tags = []
    f1_scores = []
    supports = []

    # Open the log file and read its contents
    with open(file_path, "r") as file:
        for line in file:
            # Check if line starts with "accuracy"
            if line.strip().startswith("9       "):
                # Split the line by whitespace
                parts = line.split()
                tags.append(parts[0])
                f1_scores.append(float(parts[3]))
                supports.append(int(parts[4]))

    # Create a dataframe from the extracted values
    df = pd.DataFrame({
        "tag": tags,
        "f1": f1_scores,
        "support": supports
    })

    # Using the 'bmh' style context for this plot
    with plt.style.context('bmh'):
        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(20, 14))

        # Plotting the F1 score
        ax.plot(df.index, df['f1'], label='F1 Score')  # Assuming 'f1' is the column name

        # Highlight the range from index 10 to 25
        if option1_v != 'Label Flipping Attack':
            a = int(slider4)- 10
            b = int(slider4)+ 10
            ax.axvspan(a, b, color='red', alpha=0.3)

        # Adding labels and title
        ax.set_xlabel('Round')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score by Round')
        ax.set_ylim(0,1)

        # Add a legend
        ax.legend()

        fig.savefig('./static/css/target.jpg')

def vic_f1(file_path,option1_v, option2_v, optionCI_v, slider5, slider4, slider3, range_val):

    tags = []
    f1_scores = []
    supports = []

    # Open the log file and read its contents
    with open(file_path, "r") as file:
        for line in file:
            # Check if line starts with "accuracy"
            if line.strip().startswith("1      "):
                # Split the line by whitespace
                parts = line.split()
                tags.append(parts[0])
                f1_scores.append(float(parts[3]))
                supports.append(int(parts[4]))

    # Create a dataframe from the extracted values
    df = pd.DataFrame({
        "tag": tags,
        "f1": f1_scores,
        "support": supports
    })

    # Using the 'bmh' style context for this plot
    with plt.style.context('bmh'):
        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(20, 14))

        # Plotting the F1 score
        ax.plot(df.index, df['f1'], label='F1 Score')  # Assuming 'f1' is the column name

        # Highlight the range from index 10 to 25
        if option1_v != 'Label Flipping Attack':
            a = int(slider4)- 10
            b = int(slider4)+ 10
            ax.axvspan(a, b, color='red', alpha=0.3)

        # Adding labels and title
        ax.set_xlabel('Round')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score by Round')
        ax.set_ylim(0, 1)

        # Add a legend
        ax.legend()

        fig.savefig('./static/css/vic.jpg')

#pca



@app.route('/')
def index():
    return render_template('index.html')
  
@app.route('/submitForm', methods=['POST'])  
def submitForm():
    # 从请求中获取 JSON 数据  
    data = request.get_json()
    print(data)
    option1_v = data.get('option1_v')
    option2_v = data.get('option2_v')
    optionCI_v = data.get('optionCI_v')
    slider5 = data.get('slider5')  
    slider4 = data.get('slider4')  
    slider3 = data.get('slider3')  
    range_val = data.get('range')
  
   # 假设这是我们三张图片的 URL
    print(type(option2_v))
    p_list, available_numbers = all_f1(option1_v, option2_v, optionCI_v, slider5, slider4, slider3, range_val)
    vic_f1(file_path,option1_v, option2_v, optionCI_v, slider5, slider4, slider3, range_val)
    print("done")

    target_f1(file_path,option1_v, option2_v, optionCI_v, slider5, slider4, slider3, range_val)
    print("done")


    image_url1 = './static/css/overall.jpg'
    image_url2 = './static/css/vic.jpg'
    image_url3 = './static/css/target.jpg'
    #poisoner = [1,2,3]
    #normal = [2,3,4]
    poisoner = p_list
    normal = available_numbers


    # 创建一个包含每张图片 URL 的字典  
    response = {  
        'message': 'f1 figure finish',
        'image1_url': image_url1,  
        'image2_url': image_url2,  
        'image3_url': image_url3,
        "poisoner": poisoner,
        'normal':normal


    }  

    # 返回 JSON 响应给前端  
    return jsonify(response)

@app.route('/view_visualization')
def view_visualization():
    '''fname = file_name + '_gradients.json'
    directory_path = 'data/pca/'
    file_p = os.path.join(directory_path, fname)

    gradients, worker_ids = extract_gradients(file_p)
    full_l = session.get('full_array')
    full_l = [int(i) for i in full_l]
    print(full_l)
    print(type(full_l))
    trace = session.get('radioValue')
    plot_html,image_path = visualize_3d(gradients, worker_ids,trace,POISONED_WORKER_IDS,full_l)
    '''
    plot_html = session.get('plot_html_4')
    return plot_html

@app.route('/view_visualization2')
def view_visualization2():
    plot_html_5 = session.get('plot_html_5')
    return plot_html_5

@app.route('/submit_array', methods=['POST'])  
def submit_array():
    data = request.get_json()
    array1 = data.get('array1', [])
    array2 = data.get('array2', [])
    range_val = data.get('range')
    radioValue = data.get('radioValue')

    full_array = array1 + array2

    fname = file_name + '_gradients.json'
    directory_path = 'data/pca/'
    file_p = os.path.join(directory_path, fname)

    gradients, worker_ids = extract_gradients(file_p)
    full_l = [int(i) for i in full_array]
    trace = session.get('radioValue')
    plot_html_4, image_path_4 = visualize_3d(gradients, worker_ids, trace, POISONED_WORKER_IDS, full_l)

    s_point = int(range_val[0])
    e_pont = int(range_val[1])
    print(s_point)
    print("__")
    print(e_pont)

    with open(file_p, "r") as file:
        data = json.load(file)

    # Generate the HTML for the visualization using appropriate visualization function
    plot_html_5,image_path_5 = visualize_3d_by_round_f(data, s_point, e_pont)

    session['plot_html_4'] = plot_html_4
    session['plot_html_5'] = plot_html_5

    visualization_url4 = image_path_4
    visualization_url5 = image_path_5

  
    image_url4 = './static/css/pca.JPG'
    image_url5 = './static/css/round.JPG'
  
    response = {
        'message': 'finish generation',
        'image4_url': visualization_url4,
        'image5_url': visualization_url5

    }  

    return jsonify(response)




if __name__ == '__main__':  
    app.run(debug=True)