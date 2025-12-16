def main():
    convert = {
        "I":1,
        "II":2,
        "IIi":3,
        "IV":4,
        "V":5,
        "VI":6,
        "VII":7,
        "VIII":8,
        "IX":9,
        "X":10,
        "L":50,
        "C":100,
        "D":500,
        "M":1000
    }
    word = input()
    splited = []
    sum_num = 0
    for i in range(len(word)-1):
        if convert[word[i]] < convert[word[i+1]]:
            splited.append(f"{word[i]}{word[i+1]}")
        else:
            splited.append(f"{word[i]}")
            if i == len(word)-2:
                splited.append(f"{word[i+1]}")
    
    try:
        sum_num = sum([convert[n] for n in splited])
    except KeyError and TypeError:
        for rom_num in splited:
            temp = 0
            if len(rom_num) > 1:
                temp = convert[rom_num[1]] - convert[rom_num[0]]
                sum_num += temp
            else:
                sum_num += convert[rom_num]

def convert_back(num,convert):
    thous_remin = num // 1000
    red_remin = num // 100
    ten_remin = num // 10
    remin = num % 10

    roman_num = [thous_remin,red_remin,ten_remin,remin]
    roman_num = [convert[num] for num in roman_num]
    roman = f"{roman_num[0]}M{roman_num[1]}C{roman_num[2]}X{roman_num}I"
    
main()