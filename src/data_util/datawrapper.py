from src.data.movielens100k import Movielens100k
from src.data.movielens1m import Movielens1m
from src.data.frappe import Frappe
from src.data.goodbook import GoodBook
#from src.data.shopping import Shopping
# from src.data.movielens10m import Movielens10m

class DataWrapper:

    def __init__(self, args) -> None:
        pass
        # datatype에 따른 데이터를 self.data에 저장
        if args.datatype=="ml100k":
            self.data = Movielens100k('dataset/ml-100k','u.data', args.fold)
        elif args.datatype=="ml1m":         # 수정필요
            self.data = Movielens1m(args)
        elif args.datatype=="frappe":
            self.data = Frappe(args)        # 수정완료
        elif args.datatype=="goodbook":
            self.data = GoodBook(args)
        # elif args.datatype=="shopping":     # 수정필요
        #     self.data = Shopping(args)
        else:
            raise NotImplementedError

        # 각 data마다 만들어진 class에 있는 data_getter 사용
        self.train, self.test, self.item_info, self.user_info = self.data.data_getter()

    def get_data(self):
        self.ui_matrix = self.get_ui_matrix()
        return self.train, self.test, self.item_info, self.user_info, self.ui_matrix

    
    def get_ui_matrix(self):
        train = self.train
        # 행이 user 열이 item인 관계 matrix 생성
        ui_matrix = train.pivot_table(index='user_id',columns='item_id',values='rating')
        ui_matrix = ui_matrix.fillna(0)
        ui_matrix = ui_matrix.astype(float)
        # 별점이 3 이상일 경우 1로 설정
        ui_matrix[ui_matrix >= 3] = 1
        ui_matrix[ui_matrix < 3]  = 0
        ui_matrix = ui_matrix.to_numpy()
        ui_matrix = ui_matrix.astype(float)

        return ui_matrix
    


    def get_col_type(self):
        # 범주형, 연속형을 나누어서 저장
        cat_cols = []
        cont_cols = []
        cat_cols.append('user_id')
        cat_cols.append('item_id')
        
        for col in self.item_info.columns:
            if col=='item_id':
                continue
            if self.item_info[col].dtype=='object':
                cat_cols.append(col)
            elif self.item_info[col].dtype=='int64':
                cat_cols.append(col)
            elif self.item_info[col].dtype=='float64':
                cont_cols.append(col)
    
        for col in self.user_info.columns:
            if col=='user_id':
                continue
            if self.user_info[col].dtype=='object':
                cat_cols.append(col)
            elif self.user_info[col].dtype=='int64':
                cat_cols.append(col)
            elif self.user_info[col].dtype=='float64':
                cont_cols.append(col)
    
        return cat_cols, cont_cols