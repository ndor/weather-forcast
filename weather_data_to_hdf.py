import numpy as np
import h5py, sys, os

def directories_hyrarchy_dictionary_at(path=None):
    '''
    :param path:
    :return:
    '''

    # initial access:
    if path == None:
        path = os.getcwd() # current directory
    subdirectories = os.listdir(path)
    list_of_directories = [subdirectory*os.path.isdir(path + '/' + subdirectory) for subdirectory in subdirectories]
    years = []
    for directory in list_of_directories:
        if directory != '':
            year = int(directory)
            years.append(year)
    years = np.sort(np.array(years))
    years = years.tolist()
    # print years

    # creating the dictionary:
    years = dict.fromkeys((map(str, years)), None)
    for year_key in years.keys():
        year_path = path + '/' + year_key
        subdirectories = os.listdir(year_path)
        list_of_directories = [subdirectory*os.path.isdir(year_path + '/' + subdirectory) for subdirectory in subdirectories]
        locations_list = []
        for location in list_of_directories:
            if location != '' and location != 'REPORTS':
                locations_list.append(location)
        years[year_key] = locations_list
    # print years

    # TODO: automate listing...
    sites_tag = ['SBOK', 'HZVA', 'BESR', 'BRSH', 'MIZP', 'ARAD', 'YOTV', 'SDOM', 'ELAT']
    year_tag = ['1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002']
    months_tag = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    # adding months data per site to HDF5 file:
    with h5py.File('sites_weather.hdf5', 'w') as hf:
        for site in sites_tag:
            f = []
            for year in year_tag:
                for month in months_tag:
                    if site in years[year]:
                        path_to_month = path + '/' + year + '/' + site + '/' + site + year[2:] + '.' + month
                        if os.path.exists(path_to_month):
                            k = np.loadtxt(path_to_month, skiprows=1)
                            d = ((((k[:, 1] -1) / 24).astype('int') + 1) *np.ones((1, 1))).T
                            # day number
                            d[d > 31] = 1
                            m = (k[:, 0] * np.ones((1, 1))).T
                            h = np.around((24 * (((k[:, 1]-1) / 24) - (d.T-1)))).T#.astype('int') # hour
                            # p = (months_tag.index(month) + 1) * np.ones((k.shape[0], 1)) # already exist
                            l = (int(year) * np.ones((k.shape[0], 1)))
                            f.append(np.hstack([l, m, d, h, k[:, 2:]]))
                            # print path_to_month
                            # print 'is: ' + site + ' ' + year + ' ' + month
                        # else:
                            # print 'none: ' + site + ' ' + year + ' ' + month
            # print np.array(f)

            # HDF5 data per each site:
            print 'writing ' + site + ' data to HDF5 file...'
            f = np.vstack(f)
            hf.create_dataset(site, data=f)


# TODO: dataset creator, by name of site
# TODO: net trainer (LSTM NN)
# TODO: net tester

# directories_hyrarchy_dictionary_at()
