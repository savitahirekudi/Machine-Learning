age_input=input("enter your age:")
age=int(age_input)
if age<0:
    print("enter the valid age.")
elif age<18:
    print("you are a minor")
elif age>=18 and age<65:
    print("you are a adult")
else:
    print("you are a senior citizen")     
    