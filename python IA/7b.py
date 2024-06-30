class employee:
    def __init__(self):
        self.name = ""
        self.empID = ""
        self.dept = ""
        self.salary = ""
    def getEmpDet(self):
        self.name = input("Enter The Name of the employee : ")
        self.empID = input("Enter the ID : ")
        self.dept = input("Emter the working department : ")
        self.salary = input("Enter the Salary : ")
    def showEmpDet(self): 
        print("Name : ",self.name,"\nEmployee ID : ",self.empID,"\nDepartment : ",self.dept,"\nSalary : ",self.salary)
    def updatesal(self):
        self.salary = int(input("Enter New Salary : "))
        print("Updated Salary ")
e1 = employee()
e1.getEmpDet()
e1.showEmpDet()
e1.updatesal()
e1.showEmpDet()