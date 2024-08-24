{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNTlRiCk65iBV+zNioN9wuA",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/jayuan101/Credit-Card-Approval-Prediction/blob/main/Streamlit.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Import necessary libraries\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
        "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score\n"
      ],
      "metadata": {
        "id": "cqp1ASpLbhLr"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Streamlit App Title\n",
        "st.title('Credit Card Approval Prediction')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LklVTj_6dck6",
        "outputId": "672baa23-a3fd-45c1-f020-45d9e33fa8d3"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-08-24 00:47:10.700 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator()"
            ]
          },
          "metadata": {},
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Upload CSV File\n",
        "uploaded_csv = st.file_uploader(\"data.csv\", type=\"csv\")"
      ],
      "metadata": {
        "id": "XPdwVvuCbkP5"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Navigation\n",
        "page = st.sidebar.selectbox(\"Choose a page\", [\"Analysis\", \"ML Prediction\"])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cMiGnHjdgSXH",
        "outputId": "7b81620c-87d7-4d4b-b0ec-1b929961a4f0"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-08-24 00:47:10.767 Session state does not function when running a script without `streamlit run`\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 1: Load and Display CSV\n",
        "if uploaded_csv is not None:\n",
        "    # Read the CSV into a DataFrame\n",
        "    data = pd.read_csv(uploaded_csv) # Use uploaded_csv instead of data.csv\n"
      ],
      "metadata": {
        "id": "cQnUHZR4gaP2"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Display DataFrame\n",
        "st.dataframe(df) # Use 'data' instead of 'df'"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 165
        },
        "id": "ELlmKCgny3QF",
        "outputId": "044228b2-1637-45c3-8af4-01c9bd9f4af3"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'df' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-8-9f54d32cfcca>\u001b[0m in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# Display DataFrame\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mst\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdataframe\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdf\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;31m# Use 'data' instead of 'df'\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'df' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Create a chart\n",
        "st.line_chart(df)"
      ],
      "metadata": {
        "id": "Hv4euVX9zDuf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Handle missing values (optional, based on data)\n",
        "df.dropna(inplace = True)"
      ],
      "metadata": {
        "id": "JUobR0O1tWsP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 2: Simulate Notebook Execution\n",
        "if \"Credit-Card-Approval-Prediction.ipynb\" is not None:\n",
        "    st.write(\"### Python Notebook Uploaded Successfully\")\n",
        "    st.write(\"Running notebook logic...\")"
      ],
      "metadata": {
        "id": "xLXsXvMdttpc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Split data into features and target variable\n",
        "X = df.drop(columns=['CNT_FAM_MEMBERS', 'CNT_CHILDREN' , 'FLAG_MOBIL', 'STATUS'])  # Replace 'target_column' with your actual target column name\n",
        "y = df['STATUS']"
      ],
      "metadata": {
        "id": "kPbogR-qbrih"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "r1NJzEqLaZC0"
      },
      "outputs": [],
      "source": [
        "    # Encode categorical variables if necessary\n",
        "    st.write(\"### Data Preprocessing\")\n",
        "    label_encoders = {}\n",
        "    for col in X.select_dtypes(include=['object']).columns:\n",
        "        le = LabelEncoder()\n",
        "        X[col] = le.fit_transform(X[col])\n",
        "        label_encoders[col] = le"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Scale the features\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled = scaler.transform(X_test)\n"
      ],
      "metadata": {
        "id": "G9Rdhwdbt_AH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 2: Model Training\n",
        "st.write(\"### Model Training\")\n",
        "model = RandomForestClassifier()\n",
        "model.fit(X_train_scaled, y_train)"
      ],
      "metadata": {
        "id": "0neT7ip4t_Cn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 3: Model Evaluation\n",
        "st.write(\"### Model Evaluation\")\n",
        "y_pred = model.predict(X_test_scaled)\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "st.write(f\"Accuracy of the model: {accuracy:.2f}\")"
      ],
      "metadata": {
        "id": "9BBZh-OVt_Eu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "  # Optional VIF (multicollinearity check)\n",
        "vif_data = pd.DataFrame()\n",
        "vif_data[\"feature\"] = X.columns\n",
        "vif_data[\"VIF\"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]\n",
        "st.write(\"### Variance Inflation Factor (VIF)\")\n",
        "st.dataframe(vif_data)"
      ],
      "metadata": {
        "id": "K4ygFv_UuME_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "st.write(\"### Variance Inflation Factor (VIF)\")\n",
        "st.dataframe(vif_data)"
      ],
      "metadata": {
        "id": "gpITxc0HuMHe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Show Predictions\n",
        "st.write(\"### Predictions on Test Set\")\n",
        "predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)\n",
        "st.dataframe(predictions.head())"
      ],
      "metadata": {
        "id": "3bsGo7-TuTSv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "else:\n",
        "    st.write(\"Please upload a CSV file to see the data.\")"
      ],
      "metadata": {
        "id": "4ks6CySzuTVn"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}