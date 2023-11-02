

def yes_no_prompt(msg, default=True):
    yes_answers = {"y", "yes"}
    no_answers = {"n", "no"}
    yes = 'y'
    no = 'n'
    if default is True:
        yes_answers.add("")
        yes = yes.upper()
    elif default is False:
        no_answers.add("")
        no = no.upper()

    options = f"[{yes}/{no}]"
    answer = None
    while answer not in yes_answers | no_answers:
        answer = input(f"{msg} {options} ").lower()
    return answer in yes_answers