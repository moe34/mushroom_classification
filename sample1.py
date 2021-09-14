class Human:
    #コンストラクタ
    def __init__(self, height, weight, year):
        self.height = height
        self.weight = weight
        self.year = year
    
    def calculate_BMI(self):
        bmi = self.weight / (self.height**2)
        return bmi
    
    def glow_old(self, after):
        self.year += after

class Human_detailed(Human):
    def __init__(self, height, weight, year, blood_pressure, eyesight_right, eyesight_left):
        #親クラスのコンストラクタの呼び出し
        super().__init__(height, weight, year)

        self.blood_pressure = blood_pressure
        self.eyesight_right = eyesight_right
        self.eyesight_left = eyesight_left

#インスタンス作成
human1_data_detailed = Human_detailed(175, 60, 25, 110, 1.0, 0.8)
print(human1_data_detailed.calculate_BMI())

