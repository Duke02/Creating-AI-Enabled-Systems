import numpy as np
import pandas as pd
import typing as tp
import os


class CarCsvCleaner:

    def __init__(self, data_path: str = './data/cars.csv'):
        self.data_path = data_path
        self.dataframe = None

    def load_data(self) -> pd.DataFrame:
        self.dataframe = pd.read_csv(self.data_path)
        return self.dataframe

    @staticmethod
    def _onehot_encode_column(df: pd.DataFrame, cols: tp.List[str], prefixes: tp.List[str]) -> tp.Tuple[pd.DataFrame, tp.List[str]]:
        df: pd.DataFrame = df.copy()

        new_columns: tp.List[str] = []

        for column, prefix in zip(cols, prefixes):
            dummies: pd.DataFrame = pd.get_dummies(df[column], prefix=prefix)

            new_columns.extend(dummies.columns)

            df: pd.DataFrame = pd.concat([df, dummies], axis=1)
            df: pd.DataFrame = df.drop(column, axis=1)

        return df, new_columns

    @staticmethod
    def _translate_to_english(belarus_phrase: str) -> str:
        if belarus_phrase == 'Минская обл.':
            return 'Minsk region'
        elif belarus_phrase == 'Гомельская обл.':
            return 'Gomel region'
        elif belarus_phrase == 'Брестская обл.':
            return 'Brest region'
        elif belarus_phrase == 'Могилевская обл.':
            return 'Mogilev region'
        elif belarus_phrase == 'Витебская обл.':
            return 'Vitebsk region'
        elif belarus_phrase == 'Гродненская обл.':
            return 'Grodno region'
        else:
            return 'ERROR'

    def process_columns(self) -> pd.DataFrame:
        # Translate region.
        self.dataframe['region_english'] = (self.dataframe['location_region'].map(CarCsvCleaner._translate_to_english)
                                            .str.slice(stop=-7))

        onehot_encoded_columns: tp.List[str] = ['manufacturer_name', 'region_english', 'engine_type', 'body_type', 'state']
        onehot_encoded_prefixes: tp.List[str] = ['make', 'region', 'engine type', 'body', 'state']

        self.dataframe, onehot_new_cols = CarCsvCleaner._onehot_encode_column(self.dataframe, onehot_encoded_columns, onehot_encoded_prefixes)

        self.dataframe['is_automatic'] = self.dataframe['transmission'] == 'automatic'

        scraped_year: int = 2019
        self.dataframe['age_years'] = scraped_year - self.dataframe['year_produced']
        self.dataframe['is_all_wheel_drive'] = self.dataframe['drivetrain'] == 'all'

        boolean_columns: tp.List[str] = ['engine_has_gas', 'has_warranty', 'is_exchangeable', 'is_all_wheel_drive', 'is_automatic']
        self.dataframe[boolean_columns] = self.dataframe[boolean_columns].astype(np.int8)

        return self.dataframe

    def drop_columns(self) -> pd.DataFrame:
        bad_feature_names: tp.List[str] = [f'feature_{i}' for i in range(10)]
        irrelevant_columns: tp.List[str] = ['number_of_photos', 'up_counter', 'duration_listed', 'model_name', 'color']
        processed_columns: tp.List[str] = ['location_region', 'drivetrain', 'transmission', 'year_produced']

        columns_to_drop: tp.List[str] = bad_feature_names + irrelevant_columns + processed_columns

        self.dataframe = self.dataframe.drop(columns_to_drop, axis=1)
        return self.dataframe

    def save_to_csv(self, save_path: tp.Optional[str] = None) -> str:
        if save_path is None:
            save_path: str = os.path.join('.', 'data', 'processed_cars.csv')

        self.dataframe.to_csv(save_path, index=False)

        return save_path


if __name__ == '__main__':
    print('Starting...')
    cleaner: CarCsvCleaner = CarCsvCleaner()
    print(f'Loading data from "{cleaner.data_path}"...')
    cleaner.load_data()
    print('Processing columns...')
    cleaner.process_columns()
    print('Dropping columns...')
    cleaner.drop_columns()
    print('Saving to path...')
    save_path: str = cleaner.save_to_csv()
    print(f'Saved to "{save_path}"!')

    print('Done!')

