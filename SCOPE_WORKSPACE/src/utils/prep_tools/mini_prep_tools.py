from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd
import numpy as np

def one_hot_encode_columns(data, cat_cols, case_cols, event_cols, oh_encoder_dict=None):
    case_cols_encoded = [col for col in case_cols if col not in cat_cols]
    event_cols_encoded = [col for col in event_cols if col not in cat_cols]

    if oh_encoder_dict is None:
        oh_encoder_dict = {}
        for col in cat_cols:
            oh_encoder_dict[col], data, cat_col_encoded = one_hot_encode_column(col = col, data = data)
            if col in case_cols:
                case_cols_encoded.extend(cat_col_encoded)
            elif col in event_cols:
                event_cols_encoded.extend(cat_col_encoded)
    else:
        #(oh_encoder_dict is known)
        for col, oh_encoder in oh_encoder_dict.items():
            _, data, cat_col_encoded = one_hot_encode_column(col = col, data = data, oh_encoder = oh_encoder)
            if col in case_cols:
                case_cols_encoded.extend(cat_col_encoded)
            elif col in event_cols:
                event_cols_encoded.extend(cat_col_encoded)

    return oh_encoder_dict, data, case_cols_encoded, event_cols_encoded
    
def one_hot_encode_column(col, data, oh_encoder=None):
    if oh_encoder is None:
        oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_col = oh_encoder.fit_transform(data[[col]])
        cat_col_encoded = oh_encoder.get_feature_names_out(input_features=[col])
    else:
        #(oh_encoder is known)
        encoded_col = oh_encoder.transform(data[[col]])
        cat_col_encoded = oh_encoder.get_feature_names_out(input_features=[col])
    df_enc = pd.DataFrame(encoded_col, columns=cat_col_encoded)
    data = data.reset_index(drop=True).join(df_enc)
    data.drop(columns=[col], inplace=True)

    return oh_encoder, data, cat_col_encoded

def scale_columns(data, scale_cols, case_cols, scaler_dict=None):
    if scaler_dict is None:
        scaler_dict = {}
        for col in scale_cols:
            if col in data.columns:
                scaler_dict[col], data = scale_column(col, data, case_cols)
    else:
        for col, scaler in scaler_dict.items():
            if col in data.columns:
                _, data = scale_column(col, data, case_cols, scaler)

    return scaler_dict, data


def scale_column(col, data, case_cols, scaler=None):
    non_null_col_rows = ~data[col].isnull()

    if not data.loc[non_null_col_rows, col].empty:
        data[col] = data[col].astype(float)

        if scaler is None:
            scaler = MinMaxScaler()

            if col in case_cols + ['outcome']:
                # Fit using one row per case (first occurrence)
                fit_data = (
                    data[non_null_col_rows]
                    .drop_duplicates(subset='case_nr')
                    .loc[:, [col]]
                    .values
                )
            else:
                fit_data = data.loc[non_null_col_rows, col].values.reshape(-1, 1)

            scaler.fit(fit_data)

        # Apply transform to all non-null values
        data.loc[non_null_col_rows, col] = scaler.transform(
            data.loc[non_null_col_rows, col].values.reshape(-1, 1)
        ).flatten()

    return scaler, data