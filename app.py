from flask import Flask, request, render_template
import pickle


app=Flask(__name__)
# unplickling the model

file = open('churnpredictor.pkl', 'rb')
rf = pickle.load(file)
file.close()

file1 = open('scaler.pkl', 'rb')
scaler = pickle.load(file1)
file1.close()

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        '''Gender', 'Subscription_Length_Months', 'Monthly_Bill',
       'Total_Usage_GB', 'Location_Houston', 'Location_Los Angeles',
       'Location_Miami', 'Location_New York', 'Age_Group_20-30',
       'Age_Group_30-40', 'Age_Group_40-50', 'Age_Group_50-60',
       'Age_Group_60+'''

        mydict = request.form
        gender = int(mydict['gender'])
        subscription_length_months = int(mydict['subscription_length_months'])
        monthly_bill = int(mydict['monthly_bill'])
        total_usage_gb = int(mydict['total_usage_gb'])
        location_houston = int(mydict['location_houston'])
        location_los_angeles = int(mydict['location_los_angeles'])
        location_miami = int(mydict['location_miami'])
        location_new_york = int(mydict['location_new_york'])
        age_group_20_30 = int(mydict['age_group_20_30'])
        age_group_30_40 = int(mydict['age_group_30_40'])
        age_group_40_50 = int(mydict['age_group_40_50'])
        age_group_50_60 = int(mydict['age_group_50_60'])
        age_over_60 = int(mydict['age_over_60'])

        inputfeatures = [[gender,subscription_length_months, monthly_bill, total_usage_gb,
                          location_houston, location_los_angeles,
                          location_miami, location_new_york, age_group_20_30, age_group_30_40,
                          age_group_40_50, age_group_50_60, age_over_60]]

        # predicting the class either 0 or 1
        scaled_features = scaler.transform(inputfeatures)
        predictedclass = rf.predict(scaled_features)

        print(predictedclass)
        placemap = {1: 'Will be churn', 0: 'Will not churn'}
        predictedclasssend = placemap[predictedclass[0]]

        if predictedclass[0] == 1:
            return render_template('show.html', predictedclasssend=predictedclasssend)
        else:
            return render_template('show.html', predictedclasssend=predictedclasssend)


    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
