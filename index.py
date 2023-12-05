from flask import Flask, render_template, request
import pandas as pd
import pickle as pkl


app = Flask(__name__)
car = pd.read_csv("cleaned_car.csv")

@app.route('/',methods=['GET','POST'])
def index():

    companies = sorted(car['Company'].unique())
    car_models = sorted(car['Name'].unique())
    year = sorted(car['Year'].unique(),reverse=True)
    fule_type = car['Fuel_type'].unique()
    predicted_price = None

    if request.method == 'POST':

        Company_form = request.form.get('compnany')
        car_models_form =request.form.get('car_models')
        year_form = request.form.get('year')
        fule_type_form = request.form.get('fule_type')
        kms_driven_form = request.form.get('kms_driven')
        predicted_price = predict(Company_form,car_models_form,year_form,fule_type_form,kms_driven_form)

    return render_template('index.html',companies = companies,car_models=car_models,year=year,fule_type=fule_type,predicted_price=predicted_price)

def predict(compnany,car_models,year,fule_type,kms_driven):
    model = pkl.load(open('LinearRegressionModel.pkl', 'rb'))
    data = {
        'Company':compnany,
        'Name':car_models,
        'Year': year,
        'Fuel_type':fule_type,
        'Kms_driven':kms_driven
    }
    df = pd.DataFrame([data])
    y = model.predict(df)
    return y[0]

