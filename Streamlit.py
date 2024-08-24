{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPzMXhzKmAW2c4wJcR+RFWU",
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
      "execution_count": 52,
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
        "outputId": "2dbf1d71-c324-49a3-8376-886cb80ba35d"
      },
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator()"
            ]
          },
          "metadata": {},
          "execution_count": 53
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
      "execution_count": 54,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Navigation\n",
        "page = st.sidebar.selectbox(\"Choose a page\", [\"Analysis\", \"ML Prediction\"])"
      ],
      "metadata": {
        "id": "cMiGnHjdgSXH"
      },
      "execution_count": 55,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 1: Load and Display CSV\n",
        "if uploaded_csv is not None:\n",
        "    # Read the CSV into a DataFrame\n",
        "    df = pd.read_csv(uploaded_csv)"
      ],
      "metadata": {
        "id": "cQnUHZR4gaP2"
      },
      "execution_count": 56,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Display the DataFrame\n",
        "st.write(\"### Data Preview\")\n",
        "st.dataframe(df.head())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 183
        },
        "id": "bva8I9AOg-iH",
        "outputId": "05600972-7bdd-46d3-9cf5-390aa4d93b7a"
      },
      "execution_count": 60,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'df' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-60-035226806437>\u001b[0m in \u001b[0;36m<cell line: 3>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# Display the DataFrame\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mst\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwrite\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"### Data Preview\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mst\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdataframe\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhead\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'df' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Handle missing values (optional, based on data)\n",
        "df.dropna(inplace=True)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 165
        },
        "id": "JUobR0O1tWsP",
        "outputId": "d2ae7114-8fe2-4ab4-ecca-0b310971f1cf"
      },
      "execution_count": 58,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'df' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-58-823179b08304>\u001b[0m in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# Handle missing values (optional, based on data)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mdf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdropna\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minplace\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'df' is not defined"
          ]
        }
      ]
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
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xLXsXvMdttpc",
        "outputId": "f8f66295-4729-45dd-ae45-d9c29217eecb"
      },
      "execution_count": 63,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<>:2: SyntaxWarning: \"is not\" with a literal. Did you mean \"!=\"?\n",
            "<>:2: SyntaxWarning: \"is not\" with a literal. Did you mean \"!=\"?\n",
            "<ipython-input-63-bed543eeb609>:2: SyntaxWarning: \"is not\" with a literal. Did you mean \"!=\"?\n",
            "  if \"Credit-Card-Approval-Prediction.ipynb\" is not None:\n"
          ]
        }
      ]
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
      "execution_count": 59,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 219
        },
        "id": "r1NJzEqLaZC0",
        "outputId": "e766c0f1-bb80-492f-df46-6b53c0ea4018"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'X' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-59-f8409887c57e>\u001b[0m in \u001b[0;36m<cell line: 4>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mst\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwrite\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"### Data Preprocessing\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mlabel_encoders\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0;32mfor\u001b[0m \u001b[0mcol\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mselect_dtypes\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minclude\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'object'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m     \u001b[0mle\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mLabelEncoder\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m     \u001b[0mX\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mcol\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mle\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit_transform\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mcol\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'X' is not defined"
          ]
        }
      ],
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
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 110
        },
        "id": "G9Rdhwdbt_AH",
        "outputId": "34b9f65f-f110-42e4-923f-689ee38fb46e"
      },
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "error",
          "ename": "IndentationError",
          "evalue": "unexpected indent (<ipython-input-64-ca85e22b93aa>, line 2)",
          "traceback": [
            "\u001b[0;36m  File \u001b[0;32m\"<ipython-input-64-ca85e22b93aa>\"\u001b[0;36m, line \u001b[0;32m2\u001b[0m\n\u001b[0;31m    scaler = StandardScaler()\u001b[0m\n\u001b[0m    ^\u001b[0m\n\u001b[0;31mIndentationError\u001b[0m\u001b[0;31m:\u001b[0m unexpected indent\n"
          ]
        }
      ]
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