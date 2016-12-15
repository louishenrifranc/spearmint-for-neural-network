import cPickle
import json

yes = set(['yes', 'y', 'ye', ''])
no = set(['no', 'n'])


class Parser(object):
    def __init__(self,
                 old_filename='config.pb',
                 new_filename='config_new.pb'):
        """
        Open the file and modify the config file if you have any prior about the value
        IMPORTANT :
            * name should be the first parameter of any variable in the config file
            * no comment on the size and the name lines (this is just a first version..., please :) )

        :param filename: string (default: config.pb)
            Name of the config file
        """
        self.config = open(old_filename, 'rb')
        self.new_config = open(new_filename, 'wb')
        self.priors = json.load(open('data/priors.json', 'rb'))
        self.priors = self.priors['layers']
        # While there is line to read
        self.create_lines()
        index = 0
        while True:
            # get the next line
            line = self.get_next_line(index)
            if line == None:
                break
            index += 1
            # If we read the beginning of a variable file
            if any(x in line.split() for x in ['variable', 'variable{']):
                name = ""
                while line.split()[0] != '}':
                    line = self.get_next_line(index)
                    if line == None:
                        break
                    index += 1
                    # Get the name of the variable
                    if any(x in line.split() for x in ['name', 'name=']):
                        # get its name without the \" \"
                        name = line.split()[-1][1:-1]

                    # get the size of the variable
                    if any(x in line.split() for x in ['size', 'size=']):
                        # just a check because size parameter has to come after the name variable parameter
                        self.check_if_name_is_first(name)
                        # get the size
                        size = int(line.split()[-1])
                        # modify the dictionnary, and the line
                        line = self.modify_variable(name, size, line)
                    # append a return to the file
                    if line[-1] != '\n':
                        line += '\n'
                    self.new_config.write(line)
            else:
                self.new_config.write(line)
        self.new_config.close()
        self.config.close()

    def modify_variable(self, name_var, size, line):
        # for all variable in the list
        nb_var_predefined = 0
        for layer in self.priors:
            if name_var in layer['properties']:
                nb_var_predefined += 1
        if line[-1] == '\n':
            line = line[:-1]
        nb_elements_power_of_ten = len(line.split()[-1])
        line = line[:-(nb_elements_power_of_ten)] + str(nb_var_predefined)
        return line

    def check_if_name_is_first(self, name):
        if name == "":
            raise 'please specify variable name before any parameters'

    def get_choice(self):
        choice = raw_input().lower()
        if choice in yes:
            return 1
        elif choice in no:
            return 0
        else:
            return -1

    def create_lines(self):
        self.lines = []
        for line in self.config:
            self.lines.append(line)

    def get_next_line(self, index):
        if index >= len(self.lines):
            return None
        else:
            return self.lines[index]


if __name__ == '__main__':
    parser = Parser()
