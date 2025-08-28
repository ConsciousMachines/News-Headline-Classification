import json
import tarfile

input_data = [
    {
        "inputs":"Nasdaq edges down as Nvidia falls on China market uncertainty"
    },
    {
        "inputs":"Intel CFO says chipmaker has received $5.7 billion in outstanding CHIPS Act grants under Trump deal"
    },
    {
        "inputs":"S&P 500 hits record high as Nvidia results butress AI rally"
    },
    {
        "inputs":"Nippon Steel bets on $11 billion investment, tech transfer to lift U.S. Steel profit"
    },
]

def create_json_files(data):
    for i, d in enumerate(data):
        filename = f'input{i+1}.json'
        with open(filename,'w') as f:
            json.dump(d,f, indent = 4)

def create_tar_file(input_files, output_filename = 'inputs.tar.gz'):
    with tarfile.open(output_filename, "w:gz") as tar:
        for file in input_files:
            tar.add(file)

create_json_files(input_data)
input_files = [f'input{i+1}.json' for i in range(len(input_data))]
create_tar_file(input_files)