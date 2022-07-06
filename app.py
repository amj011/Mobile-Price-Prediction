from flask import Flask, render_template, request
import pickle
import sklearn
app = Flask(__name__)
model = pickle.load(open('dtree.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    # return render_template('index2.html', cities=cities)
    return render_template('index2.html')

@app.route('/index', methods=['GET'])
def index():
    # return render_template('index.html', cities=cities)
    return render_template('index.html')

@app.route('/readme', methods=['GET'])
def readme():
    # return render_template('readme.html', cities=cities)
    return render_template('readme.html')

# cities = ["Bengaluru", "Chennai", "Faridabad", "Ghaziabad", "Gurgao", "Hyderabad", "Kolkata", "Lucknow", "Mumbai", "New Delhi", "Noida", "Pune"]

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Battery = int(request.form.get('Battery'))
        Blue = int(request.form.get('Blue'))
        Clock = float(request.form.get('Clock'))
        Sim = int(request.form.get('Sim'))
        Fc = int(request.form.get('Fc'))
        FourG = int(request.form.get('FourG'))
        Memory = int(request.form.get('Memory'))
        Dep = float(request.form.get('Dep'))
        Weight = int(request.form.get('Weight'))
        Cores = int(request.form.get('Cores'))
        Pc = int(request.form.get('Pc'))
        Height = int(request.form.get('Height'))
        Width = int(request.form.get('Width'))
        Ram = int(request.form.get('Ram'))
        H = int(request.form.get('H'))
        W = int(request.form.get('W'))
        TT = int(request.form.get('Talktime'))
        ThreeG = int(request.form.get('ThreeG'))
        Touch = int(request.form.get('Touch'))
        Wifi = int(request.form.get('Wifi'))
        
        input = [[Battery, Blue, Clock, Sim, Fc, FourG, Memory, Dep, Weight, Cores, Pc, Height, Width, Ram, H, W, TT, ThreeG, Touch, Wifi]]
        print(input)
        

        prediction = model.predict(input)
        output = prediction[0]
        print(output)
        if output == 0:
            return render_template('index.html', prediction_text="You can get the mobile between at ₹10000-15000")
        elif output == 1:
            return render_template('index.html', prediction_text="You can get the mobile between at ₹15000-20000")
        elif output == 2:
            return render_template('index.html', prediction_text="You can get the mobile between at ₹20000-25000")
        elif output == 3:
            return render_template('index.html', prediction_text="You can get the mobile between at ₹25000-30000")
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)