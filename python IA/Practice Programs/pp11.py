class employee:
    def __init__(self):
        self.name = ""
        self.EmpID = ""
        self.dept = ""
        self.salary = ""
    def getempDet(self):
        self.name = input("Enter Emp name : ")
        self.EmpID = input("Enter ID :")
    def showEmpDet(self):
        print("Name :",self.name,"\nEMP ID: ",self.EmpID)
    def updatesal(self):
        self.EmpID = input("Enter new ID: ")
e1 = employee()
e1.getempDet()
e1.showEmpDet()
e1.updatesal()
e1.showEmpDet()