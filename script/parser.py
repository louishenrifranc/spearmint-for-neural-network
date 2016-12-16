import json
import os
from utils import get_relative_filename


class Parser(object):
    """
    Parse config.pb and modify it with respect to the value in the predefined_values.json
    """

    def __init__(self,
                 old_filename='config.pb',
                 new_filename='config_new.pb',
                 write_over=True):
        """
        Parse the file layer_parameters.json and modify the config.pb required for Spearmint software

        Parameters
        ----------
        :param oldfilename: string (default: config.pb)
            Name of the old config file
        :param newfilename: string (default: config_new.pb')
            Name of the new config file
        :param write_over: boolean (default: True)
            Write over the old filename directly
        """

        # Open old config file, and the new file
        self.config = open(get_relative_filename(old_filename), 'rb')
        self.new_config = open(get_relative_filename(new_filename), 'wb')
        # Prior hardcoded information about the layers into this file
        self.priors = json.load(open(get_relative_filename('data/predefined_values.json'), 'rb'))
        self.priors = self.priors['layers']

        # copy all lines from the config.pb in a buffer
        self.create_lines()
        index = 0

        # while there is a line to read
        while True:
            # read the next line
            line = self.get_next_line(index)
            # no more line to read
            if line == None:
                break
            index += 1

            # If we read the beginning of a variable definition, then we will go over the name and the size field,
            #  and modify them in consequence of what is written in json file
            if any(x in line.split() for x in ['variable', 'variable{']):
                name = ""
                self.new_config.write(line)

                while line.split()[0] != '}':
                    line = self.get_next_line(index)
                    if line == None:
                        break
                    index += 1

                    # Get the name of the variable first
                    if any(x in line.split() for x in ['name', 'name=']):
                        # get its name without the \" \"
                        name = line.split()[-1][1:-1]

                    # get the number of parameter in the variable
                    if any(x in line.split() for x in ['size', 'size=']):
                        # just a check because size parameter has to come after the name variable parameter
                        self.check_if_name_is_first(name)
                        # get the size
                        size = int(line.split()[-1])
                        # check if the user has no forgot to define a layer. Raise only warning
                        self.check_if_nb_layers_match(size)
                        # modify the dictionnary, and the line
                        line = self.modify_line(name, size, line)
                    # append a return to the file
                    if line[-1] != '\n':
                        line += '\n'
                    self.new_config.write(line)
            else:
                self.new_config.write(line)
        self.new_config.close()
        self.config.close()

        if write_over:
            temp_file = 'temp_file_name'
            os.rename(get_relative_filename(new_filename), get_relative_filename(temp_file))
            os.rename(get_relative_filename(old_filename), get_relative_filename(new_filename))
            os.rename(get_relative_filename(temp_file), get_relative_filename(old_filename))
            os.remove(get_relative_filename(new_filename))

    def modify_line(self, name_var, size, line):
        """
        Modify a line of the config.pb file
        :param name_var: string
            Name of the actual variable modified in the config.pb file
        :param size: int
            Previous size parameter
        :param line: string
            Line to modify
        :return: the new line updated with the good size
        """
        nb_var_predefined = 0
        # Iterate over all layers
        for layer in self.priors:
            # If the hyperparameter has an entry in the json file,, then spearmint don't need to look after it
            if name_var in layer['properties']:
                nb_var_predefined += 1
        if line[-1] == '\n':
            line = line[:-1]
        nb_elements_power_of_ten = len(line.split()[-1])
        line = line[:-(nb_elements_power_of_ten)] + str(size - nb_var_predefined)
        return line

    def check_if_nb_layers_match(self, size):
        """
        Check if the number of layers match: as much layers defined in the json file, and in spearmint config file
        :param size:
        :return:
        """
        if size > len(self.priors):
            raise UserWarning(
                'More layers defined in config.pb than in predefined_values.json. \
                All undefined layers in predefined_values.json will take default parameter')
        elif size < len(self.priors):
            raise UserWarning('More layers defined in predefined_values.json than in config.pb \
                              Is this truly what you want? ')

    def check_if_name_is_first(self, name):
        """
        Check if, in the config file, the name has been set before the size.
        I did this, because I am reading the config file from top to bottom only once.
        :param name: Name of the file
        :return:
        """
        if name == "":
            raise SyntaxError('For every variable in config.pb, set the name before the size')


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
