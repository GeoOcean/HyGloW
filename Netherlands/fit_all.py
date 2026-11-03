from utils import metamodel as mm

if __name__ == "__main__":
    res = mm.fit_resources()  # prints the machine, the fits to do, the proposal
    res.workers = 24  # override the number of workers to use
    print(res)
    mm.fit_all(resources=res)
