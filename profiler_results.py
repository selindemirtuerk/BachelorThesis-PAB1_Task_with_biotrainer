import pstats
from pstats import SortKey

def prof_to_txt(prof_file, txt_file):

    with open(txt_file, 'w') as stream:
        stats = pstats.Stats(prof_file, stream=stream).sort_stats('tottime')
        stats.print_stats()
    exit(0)

def txt_to_prof(txt_file):
    try:
        with open(txt_file, "r") as f:
            data = f.read()
            stats = pstats.Stats(data)  # Attempt to create Stats object
            stats.dump_stats("profile.prof")
            print(".prof file created successfully!")
    except Exception as e:
        print(f"Error creating .prof file: {e}")



if __name__ == '__main__':
    #prof_file = ""
    prof_file = "/home/selindemirturk/PAB1_GFP_task_with_biotrainer/profile_original_EvoPlay_with_deepcopies.prof"
    txt_file = "/home/selindemirturk/PAB1_GFP_task_with_biotrainer/profile_original_EvoPlay_with_deepcopies.txt"
    prof_to_txt(prof_file, txt_file)