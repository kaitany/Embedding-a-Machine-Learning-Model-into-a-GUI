import streamlit as st
import pandas as pd
import pickle
import os
import datetime
from sklearn.preprocessing import OneHotEncoder



#setup
#variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, "best_model", "ml.pkl")

#useful functions
st.cache_resource()
def  load_ml_components(fp):
    "load the ml components to re-use in app"
    with open(fp, 'rb') as file:
        obj = pickle.load(file)
        return obj
    
#Execution
ml_components_dict = load_ml_components(fp = ml_core_fp)
preprocessor = ml_components_dict["pipeline"]

# Specify the path to your saved model
dt_model_filename = 'sales_dt_model.pkl'
sales_dt_path = os.path.join(DIRPATH, dt_model_filename)

# Load the model
model = ml_components_dict["models"]

#Define app sections
header = st.container()
dataset = st.container()
prediction = st.container()

#set up the "header" section
with header:
    header.markdown("#### **This app is to predict store sales for Corporation Favorita**")
    header.markdown("---")

# Set up the "Dataset" section
train_data_fp = os.path.join(DIRPATH, "Dataset", "final_clean_train.csv")
loaded_df = pd.read_csv(train_data_fp)

#Dropped columns not used in training
train_data = loaded_df.drop(columns=["sales", "transferred", "day_of_month"])
#Renamed differenced sales("sales_diff") to "sales"
train_data = train_data.rename(columns={"sales_diff" : "sales"})

with dataset:
    if st.checkbox("Preview the dataset"):
        # Display the first 5 rows of the DataFrame
        st.dataframe(train_data.head(10))

#create key lists
expected_inputs = ['date', 'store_nbr', 'Product', 'onpromotion', 'Oil_prices',
                    'city', 'state', 'year', 'month', 'day_of_week',]

numerical_features = ['date', 'store_nbr', 'onpromotion', 'Oil_prices', 'year', 'month']

categorical_features = ['Product', 'city', 'state', 'day_of_week']

#set up the "Prediction" section
with prediction:
    prediction.subheader("Inputs")
    prediction.write("This section will receive your inputs")
    left_col, right_col = prediction.columns(2)

# Define product list
Product_list = ['CLEANING', 'BREAD/BAKERY', 'SCHOOL AND OFFICE SUPPLIES',
       'HOME AND KITCHEN II', 'MAGAZINES', 'MEATS', 'SEAFOOD',
       'HOME AND KITCHEN I', 'PLAYERS AND ELECTRONICS', 'BEVERAGES',
       'BOOKS', 'PERSONAL CARE', 'LADIESWEAR', 'PREPARED FOODS', 'EGGS',
       'PET SUPPLIES', 'BABY CARE', 'HARDWARE', 'HOME CARE', 'GROCERY I',
       'LIQUOR,WINE,BEER', 'GROCERY II', 'LINGERIE', 'CELEBRATION',
       'POULTRY', 'DELI', 'HOME APPLIANCES', 'PRODUCE', 'LAWN AND GARDEN',
       'AUTOMOTIVE', 'BEAUTY', 'DAIRY', 'FROZEN FOODS']

#Define City list
city_list = ['Manta', 'Guayaquil', 'Quito', 'Cuenca', 'Santo Domingo', 'Ibarra',
       'Machala', 'Latacunga', 'Puyo', 'Babahoyo', 'Quevedo', 'El Carmen',
       'Salinas', 'Libertad', 'Playas', 'Ambato', 'Guaranda', 'Cayambe',
       'Esmeraldas', 'Riobamba', 'Loja', 'Daule']

#Define State list
state_list = ['Manabi', 'Guayas', 'Pichincha', 'Azuay',
       'Santo Domingo de los Tsachilas', 'Imbabura', 'El Oro', 'Cotopaxi',
       'Pastaza', 'Los Rios', 'Santa Elena', 'Tungurahua', 'Bolivar',
       'Esmeraldas', 'Chimborazo', 'Loja']

# Month names and their corresponding numerical values
month_mapping = {
    "JAN.": 1,
    "FEB.": 2,
    "MAR.": 3,
    "APR.": 4,
    "MAY": 5,
    "JUN.": 6,
    "JUL.": 7,
    "AUG.": 8,
    "SEPT.": 9,
    "OCT.": 10,
    "NOV.": 11,
    "DEC.": 12
}

# Define the form
form = st.form(key="Information", clear_on_submit=True)

#set up the form
with form:
    #Left column
    #created an empty "date" column to handle errors with preprocessor that expected a "date" column used during training.
    date = [""]
    left_col.write("[Inputs] Part A:")
    year = left_col.number_input("Enter a year", min_value = 2013, step = 1) 
    # Radio button for month selection
    selected_month_name = left_col.radio("Select month", ["JAN.", "FEB.", "MAR.", "APR.", "MAY", "JUN.", 
                                               "JUL.", "AUG.", "SEPT.", "OCT.", "NOV.", "DEC."], index=0, format_func=lambda x: x, horizontal=True)
    # Map the selected month name to its numerical value
    selected_month_value = month_mapping[selected_month_name]
    #selected month value for preprocessing
    month = selected_month_value
    
    day_of_week = left_col.radio("Select Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 
                                'Thursday', 'Friday', 'Saturday', 'Sunday'], index=1, horizontal=True)

    store_nbr = left_col.select_slider("Select a store number", options=range(1, 55), value=1)

    #Right column
    right_col.write("[Inputs] Part B:")
    Product = right_col.selectbox("Select product category", options= Product_list)
    onpromotion = right_col.radio("Is the product on promotion (1) or not (0)?", options=[1, 0], horizontal = True)
    Oil_prices = right_col.number_input("Enter Oil prices")
    city = right_col.selectbox("Select City", options = city_list)
    state = right_col.selectbox("select State", options= state_list)

    #Enter "submit" button
    submitted = form.form_submit_button("Submit")

#Upon submission
if submitted:
    with prediction:
        # Format inputs:
        input_dict = {
            "store_nbr": [store_nbr],
            "Product": [Product],
            "onpromotion": [onpromotion],
            "Oil_prices": [Oil_prices],
            "city": [city],
            "state": [state],
            "year": [year],
            "month": [month],
            "day_of_week": [day_of_week],
            #"date": [date]
        }
        input_dict["date"] = [date]

        # Convert to a DataFrame
        input_datadf = pd.DataFrame.from_dict(input_dict)
        
        #Apply preprocessor to input data
        transformed_input_data = preprocessor.transform(input_datadf)
        
        # Create DataFrame from the transformed data (dense matrix)
        input_data = pd.DataFrame(transformed_input_data)

        # Define the column names
        numerical_features = ['date','store_nbr', 'onpromotion', 'Oil_prices', 'year', 'month', ]
        categorical_features = ['Product', 'city', 'state', 'day_of_week']

        # Add the column names to the categorical columns of the DataFrame
        if hasattr(preprocessor.named_transformers_['cat']['onehot'], 'get_feature_names_out'):
            # Get the one-hot encoded column names
            cat_columns = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
            #Combine the categorical and numerical column names

            all_columns = list(cat_columns) + numerical_features 
            input_data.columns = all_columns

            #Dropped the empty "sales" column needed only for preprocessing

            input_data = input_data.drop(columns="date")

            #Make the prediction
            model_output = model.predict(input_data)  

    st.success(f"Sales: {model_output.round(2)}")
