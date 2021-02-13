
from functools import reduce
import os, sys, re, shutil, warnings, fnmatch
import numpy as np
import pandas as pd
import deepdish as dp
from pathlib import Path

class basics(object):
    _labutilspath = None
    _rock_basics_flag = True
    debug  = True

    probe_settings = {
        'perm':{
            'usecols':[0,1,2,6,12],
            'skiprows':7,
            'col':['x','y'] + ['perm'],
            'names':['x','y','perm','meas_code','tile'],
            'tip':['perm'],
            'h':3,
            'hi':2,
            'limits':[0., 1.0e4]
        },
        
        'impl':{
            'usecols':[0,1,2,3],
            'skiprows':7,
            'col':['x','y'] + ['e_star'],
            'names':['x','y','e_star','tile'],
            'tip':['e_star'],
            'h':3,
            'hi':2,
            'limits':[0., 1.0e3]
        },
        
        'vels':{
            'usecols':[0,1,2,3,6,9],
            'skiprows':7,
            'col':['x','y'] + ['vp0', 'vs0', 'vp90', 'vs90'],
            'names':['x','y','angle','vp','vs','tile'],
            'tip':['vp','vs'],
            'h':5,
            'hi':3,
            'limits':[0.5e3, 8.0e3]
        },
        
        'ftir':{
            'usecols':None,
            'skiprows':2,
            'col': ['x','y'] + ['l_'+str(int(x)) for x in np.linspace(1,1752,1752)],
            'names':['x','y'] + ['l_'+str(int(x)) for x in np.linspace(1,1752,1752)],
            'tip':['l_'+str(int(x)) for x in np.linspace(1,1752,1752)],
            'h':None,
            'hi':2,
            'lambdas': np.load(os.path.join(os.path.dirname(__file__), 'ftir-lambdas.npy')),
            'limits':[0.0, 6.0]
        }
        
    }
    
    _alphabet = {chr(x) : k + 4 for k, x in enumerate(range(97,123))}
    _alphabet[0] = 0
    _alphabet['line'] = 1
    _alphabet['zbottom'] = _alphabet['bottom'] = 2
    _alphabet['ztop'] = _alphabet['top'] = 3
    _alphabet['points'] = 30
    _alphabet_max = 30

    def vel_agg(self, x):
        x = np.mean(x)
        return x

    def _val_replace(self, v):
        try:
            val = self._alphabet[v]
        except KeyError:
            self._alphabet_max += 1
            self._alphabet[v] = self._alphabet_max
            val = self._alphabet_max
        return val
    
    def duplicated_varnames(self, df):
        """[summary]
        Return a dict of all variable names that are duplicated in the dataframe.
        source: https://stackoverflow.com/questions/26226343/pandas-concat-yields-valueerror-plan-shapes-are-not-aligned
        Args:
            df (dataframe): dataframe
        Returns:
            dict: dictionary with repeated values
        """
    
        repeat_dict = {}
        var_list = list(df) # list of varnames as strings
        for varname in var_list:
            # make a list of all instances of that varname
            test_list = [v for v in var_list if v == varname] 
            # if more than one instance, report duplications in repeat_dict
            if len(test_list) > 1: 
                repeat_dict[varname] = len(test_list)
        return repeat_dict
    
    def _find_repeated_index_values(self, fdesc, debug = False, key = 'probe'):
        potential_problem = []
        for r in fdesc.index.unique():
            test_out = reduce(lambda x, y: x == y, 
                            (np.sum(fdesc.loc[[r], key].values == v) for v in fdesc.loc[[r], key].unique()))
            if not test_out:
                potential_problem.append(r)
        if debug: print(potential_problem, sep = '\n')
        return potential_problem

    def _fix_repeated_probes(self, fdesc, fix = True, debug = False, key = 'probe', redundacy = True, apply_func = lambda s: s.split('/')[-2]):
        
        potential_problem = self._find_repeated_index_values(fdesc, debug = debug, key = key)

        if np.logical_and(fix, len(potential_problem)>0):
            index_cols = fdesc.index.names
            summary = fdesc.reset_index(drop = False).copy()
            for nproblem in potential_problem:
                pb = summary.query("tag == '%s' & subtag == '%s' & instance == '%s'" % (nproblem[2:5])).loc[:, [key]]
                summary.loc[pb.index, 'instance'] = summary.loc[pb.index, 'instance'] + '-' + summary.loc[pb.index, 'relroot'].apply(apply_func)
            
            fdesc = summary.set_index(index_cols).copy()
        
            if np.logical_and(redundacy, len(self._find_repeated_index_values(fdesc, debug = True)) == 0): print('all good')
        
        return fdesc

    def _list_concat(self, list_of_lists):
        """[summary]
        concatenate a list of lists, and flatten the output.
        Args:
            list_of_lists (list): list of lists

        Returns:
            list: flatten list with all elements in list of lists concatenated.
        """
        return [item for sublist in list_of_lists for item in sublist]
    
    def _get_rock_basics(self):
        from _helpers.basics import rock_info
        self.rock_info = rock_info()
        self._rock_basics_flag = False
        return
    
    def _get_rock_dict(self):
        if self._rock_basics_flag:
            self._get_rock_basics()
        
        rock_dict = self.rock_info.rock_dict
        return rock_dict
    
    def _get_rockinfo(self, x, key = None):
        if self._rock_basics_flag:
            self._get_rock_basics()
        s = self.rock_info.rock_dict[x.split('_')[0]][key]
        return s
    
    def _vel_direction(self, x):
        d = -1
        if x == 'velax':
            d = 1
        return d
    
    def _set_nonevar(self, var, val = None):
        if val is not None:
            self.__setattr__(var, val)
        return
    
    def _set_labutilspath(self, labutilspath):
        self.__setattr__('_labutilspath', labutilspath)
        sys.path.append(self._labutilspath)
        return
    
    def _get_labutilspath(self):
        currpath = os.path.dirname(__file__)
        labutilspath = str(Path(currpath).parents[0])
        return labutilspath

    def _fix_xy(self, xy, div = 0.5, decimals = 1, around = True, recenter = True):
        if around:
            xy = np.around(xy, decimals = decimals)
        if recenter:
            xy = div*np.divmod(xy, div)[0]
        return xy
    
    def _enforce_float(self, df):
        df = df.apply(pd.to_numeric,errors = 'coerce')#.dropna(thresh = 'all')
        df.replace(np.inf, np.nan, inplace = True)
        df.dropna(inplace = True, how = 'all')
        # reset index 
        df.reset_index(inplace = True, drop = True)

        return df

    def get_info_and_data_rows(self, file, n_max_info = 2, info_rows = False):
        with open(file, 'r') as temp_f:
            col_count = 0
            n = 0
            t = str
            d = []
            while t != float :
                l = temp_f.readline()
                c = l.split(",")
                try:
                    t = type(float(c[0]))
                except:
                    t = str
                col_count = len(c)
                n += 1
                if n == 20:
                    t = float
                d.append([col_count, n])
            if n < 20:
                n_rows_skip = n - 1
                d = np.array(d)
                n_info_rows = sum(d.T[0, :] <= n_max_info)
            else:
                n_rows_skip = -1
                n_info_rows = -1
            # if info_rows:
            #     d = np.array(d)
            #     n_info_rows = sum(d.T[0, :] <= n_max_info)
            #     return n_rows_skip, n_info_rows
            # else:
            #     return n_rows_skip
        return int(n_rows_skip), int(n_info_rows)

    def read_data(self, fpath, probe, zero_offset = True, fix_xy = True, only_data_cols = False, enforce_float = True, infer_row_settings = False, file_info = False, **kwargs):
        
        opts = {
                'usecols' : self.probe_settings[probe]['usecols'],
                'skiprows' : self.probe_settings[probe]['skiprows'],
                'names' : self.probe_settings[probe]['names']
            }
        
        if np.logical_or(infer_row_settings, file_info):
            n_rows_skip, n_info_rows = self.get_info_and_data_rows(fpath)
            if n_rows_skip == -1:
                file_info = False
                infer_row_settings = False
        
        if infer_row_settings:
            # print('inferring row settings')
            opts['skiprows'] = n_rows_skip
            
        df = pd.read_csv(fpath, **opts, **kwargs)

        if enforce_float:
            df = self._enforce_float(df)

        if fix_xy:
            df.loc[:,['x', 'y']] = df.loc[:,['x', 'y']].apply(self._fix_xy)
        
        # reset x,y offset to zero
        if zero_offset:
            df.iloc[:, :2] = df.iloc[:, :2] - df.iloc[:, :2].min()
        
        if only_data_cols:
            h = self.probe_settings[probe]['h']
            df = df.iloc[:, :h].copy()
        
        if file_info:
            df_info = pd.read_csv(fpath, nrows = n_info_rows, names = ['info', 'val'], header = None)
            df_info = df_info.pivot_table(columns = 'info', aggfunc = lambda x: x).reset_index(drop = True).drop(columns = 'File')
            return df, df_info
        else:
            return df
    
    def _save(self, data, method= 'pandas', savefile = 'data', **kwargs):
        if method =='pandas':
            data.to_csv(savefile + '.csv', index = False, **kwargs)
        if method=='h5' or method=='deepdish':
            dp.io.save(savefile + '.h5', data)
        return
    
    def save_data(self, df, method = 'pandas', savepath = './', savename = 'data', save_xyp = False, probe = None, **kwargs):
        savefile = os.path.join(savepath, savename)
        if save_xyp and (probe is not None):
            h = self.probe_settings[probe]['h']
            df = df.iloc[:, :h].copy()
        self._save(df, savefile = savefile, method = method, **kwargs)
        return
    
    def __init__(self, labutilspath = None):
        if labutilspath is None:
            labutilspath = self._get_labutilspath()
        self._set_labutilspath(labutilspath)
        return

class file_sorter(basics):
    dryrun = True

    datapath = {
        'path':None,
        'processed':'processed',
        'raw':'raw',
        'generic':'_generic',
        'fluids':'_fluids',
        'analysis':'_analysis',
        'files':{
            'exclude':['.*','_*','*.asd','*.tcl', 'summary*','*map*', 'full*','combined*'],
            'include':['*.csv']
        }
    }

    _df_colnames_ordered = ['sample_tag', 'subsample_tag', 'sample_code', 'sample_family', 
                            'probe', 'side', 'instance', 'experiment',
                            'fname', 'relroot']

    def _get_exclude_list(self):
        self.datapath['exclude'] = ['_special-studies', 'special_studies', '_special_studies', '_unsorted', 
                                    self.datapath['raw'], self.datapath['generic'], self.datapath['analysis'],
                                    '*layout*', '_postprocessed']
        self._files_excludes = r'|'.join([fnmatch.translate(x) for x in self.datapath['files']['exclude']]) or r'$.'
        self._files_includes = r'|'.join([fnmatch.translate(x) for x in self.datapath['files']['include']]) or r'$.'
        self._datapath_length = int(len(self.datapath['path'].split('/')))
        return
    
    def _rename_file(self, root, fname_old, fname_new, debug = False):
        oldname = os.path.join(root, fname_old)
        newname = os.path.join(root, fname_new)
        if debug: print("renaming:\n%s\nto\n%s" % (oldname, newname))
        shutil.move(oldname, newname)
        return

    def _get_refindall(self, pattern, string):
        v = None
        s = re.findall(pattern, string)
        if len(s)>0:
            v = s[0]
        return v

    def _get_sides(self, x):
        side = self._get_refindall(r'_([a-z0-9]+).csv$', x.lower())
        if side is not None:
            side = side[-1]
        return side

    def _get_subsample(self, x):
        sub = self._get_refindall(r'.*sub[a-z]+[/]([a-z]+|[0-9]+|[a-z]+[0-9]+|[a-z]+[0-9]+[_]+[a-z]+|[a-z]+[0-9]+[_]+[a-z]+[0-9]+)[/]',x)
        return sub

    def _get_probename(self, x):
        probe = self._get_refindall(r'(perm|vels|impl|ftir)', x)
        return probe

    def _get_instance(self, x):
        instance = self._get_refindall(r'(before|after)', x)
        return instance

    def add_before_fname(self, fname, root):
        name2 = re.sub(r'(perm|vels|impl|ftir).*(-|_)([a-z0-9]+)[.]([a-z]+)', r'\1_before_\3.\4', fname)
        if self.debug:
            print(root, fname, name2, sep='\t')
        if not self.dryrun:
            self._rename_file(root, fname, name2)
        return name2

    def swap_instance_fname(self, fname,root):
        name2 = re.sub(r'(perm|vels|impl|ftir)(_|-)([a-z]+|[0-9]+)(_|-)(.*)(before|after)(.*)[.]([a-z]+)',
                    r'\1_\6_\3.\8',
                    fname)
        if self.debug:
            print('swaping', root, fname, name2, sep='\t')
        if not self.dryrun:
            self._rename_file(root, fname, name2)
        return name2

    def check_autoscan_fname(self, fname, root):
        if (not 'before' in fname) and (not 'after' in fname):
            warnings.warn('before or after not found in ' + 
                        root.split('/')[self._datapath_length - 1] + fname)
            fname = self.add_before_fname(fname, root)
        instance = re.findall('before|after', fname)
        if len(instance)==1:
            
            instance = instance[0]
            if self.debug: print(root.split('/')[self._datapath_length - 1], instance, fname, sep='\t')
            if len(fname.split('_'))>2:
                tst = re.match(r'(perm|vels|impl|ftir)(-|_)(before|after)(-|_)([a-z0-9]+)[.]([a-z]+)', fname)
                if tst is None:
                    fname = self.swap_instance_fname(fname, root)
        else:
            if len(instance)>1:
                warnmsg = fname + ' has more than one instance: ' + ', '.join(instance)
                warnmsg = warnmsg + ' and cannot choose! \n check ' + os.path.join(root,fname) 
            if len(instance)==0:
                warnmsg = fname + ' not in either category (!) \n please review!'
            warnings.warn(warnmsg)
        return fname
    
    def wrangle(self, save = False, link = False, savepath = None, experiment = True):
        self._get_exclude_list()

        fs = []
        rs = []
        for root, dirs, files in os.walk(self.datapath['path']):
            [dirs.remove(d) for d in list(dirs) if d in self.datapath['exclude']]
            files = [f for f in files if not re.match(self._files_excludes, f)]
            files = [f for f in files if re.match(self._files_includes, f)]
            for fname in files:
                fname = self.check_autoscan_fname(fname, root)
                fs.append(fname)
                rs.append(os.path.relpath(os.path.join(root, fname), start = self.datapath['path']))

        df = pd.DataFrame({'fname':fs, 'relroot':rs})
        df['sample_tag']    = df['relroot'].apply(lambda x: x.split('/')[0])
        df['subsample_tag'] = df['relroot'].apply(self._get_subsample)

        df = pd.concat([df, 
                        df.fname.apply(lambda s: pd.Series({'probe':self._get_probename(s), 
                                                            'side':self._get_sides(s), 
                                                            'instance':self._get_instance(s)})),
                ],
                axis = 1, sort = False)

        for s in ['code', 'family']:
            df['sample_' + s] = df['sample_tag'].apply(self._get_rockinfo, key = s)

        # add experiment type 
        if experiment:
            ix = df.instance.values == 'after'
            df.loc[ix, 'experiment'] = df.loc[ix, 'relroot'].apply(lambda x: 
                re.findall(r'exp[a-z]+[/](\w+)[/]', x)[0]
                )

        df = df.loc[:, self._df_colnames_ordered]

        if (save or link) and (savepath is None):
            savepath = self.datapath['path']
        # create link column and save
        if save:
        # save all columns except link
            df.to_csv(os.path.join(savepath,'summary.csv'), index = False)
        
        if link:
            df['link'] = df['relroot'].apply(lambda x: '<a href="./{0}">link</a>'.format(x))
            df.to_html(os.path.join(savepath,'autoscan_data.html'), na_rep='-', escape=False)

        return df.loc[:, self._df_colnames_ordered]
    
    def __init__(self, datapath = None, labutilspath=None):
        super().__init__(labutilspath=labutilspath)
        if datapath is not None:
            self.datapath['path'] = datapath
        return

class postprocess(basics):
    postprocessed_folder = '_postprocessed'
    outpath = None
    outname = None
    sample_info = None
    
    info_columns = ['side', 'sample_code', 'sample_family', 'sample_tag', 'subsample_tag']
    
    info_columns_short = {
        'side':'side',
        'sample_code':'code',
        'sample_family':'family',
        'sample_tag':'tag',
        'subsample_tag':'subtag'
        }
    
    @property
    def probes(self):
        """[summary]
        self.probes
        Returns:
            list: sorted probe names
        """
        return sorted(self.probe_settings.keys())
    
    @property
    def empty_dataframe(self):
        # import copy
        # ps = copy.deepcopy(self.probe_settings)
        # # ps['vels']['names'] = [s + str(k) for k in [0, 90.0, 45.0] for s in ['vp', 'vs']]
        # ps['vels']['h'] = None
        # ps['vels']['hi'] = 0
        # cols = self._list_concat((ps[key]['names'][ps[key]['hi']:ps[key]['h']] for key in self.probes))
        df = pd.DataFrame(columns = ['x', 'y'])# + cols)
        return df
    
    def _set_outpath(self, root = './', subfolder = '', mkdir = True):
        outpath = os.path.join(root, subfolder, self.postprocessed_folder)
        if mkdir and (not os.path.exists(outpath)):
            os.mkdir(outpath)
        self.outpath = outpath
        return outpath
    
    def _set_outfilename(self, *args, joiner = '_'):
        args = list(filter(None, args))
        outname = joiner.join(args)
        self.outname = outname
        return outname
    
    def _get_info_dict(self, ds, shorthand = True):
        out = {}
        for var in self.info_columns:
            varout = var
            if shorthand:
                varout = self.info_columns_short[var]
            if self.debug: 
                print(var, varout, sep = '\t')
            out[varout] = getattr(ds, var)
        return out

    def get_sample_info(self, ds, shorthand = True):
        self.sample_info = self._get_info_dict(ds, shorthand = shorthand)
        return
    
    def pre_pool_data(self, df, run_info, probe):
        data = {}
        
        for x in self.info_columns:
            data[self.info_columns_short[x]] = run_info[x]

        for col in self.probe_settings[probe]['names'][2:]:
            data[col] = df[col].values

        df_out = pd.DataFrame(data)
        df_out = df_out.loc[:, self.probe_settings[probe]['names'][2:] + 
                                        ['side', 'code', 'family', 'tag', 'subtag']]

        return df_out
        
    def slice_data(self, df, run_info, probe = None):
    
        h   = self.probe_settings[probe]['h']
        tip = self.probe_settings[probe]['tip']

        # get the slices and check which is the middle one
        if (len(df.loc[:,'y'].unique())>1 and len(df.loc[:,'x'].unique())>1):
            slice_along = np.int(df['x'].max()/df['y'].max() >= 1)
            u = ['x','y'][slice_along]
            v = ['x','y'][np.int(not np.bool(slice_along))]
            slices_ini  = df.loc[:, u].unique()
            slices_ini.sort()
            if slices_ini.size > 1:
                median = slices_ini[slices_ini >= slices_ini.max()/2][0]
            else:
                median = slices_ini[0]
        else:
            if len(df.loc[:,'y'].unique())==1: 
                u = 'y'
                v = 'x'
            else:
                u = 'x'
                v = 'y'
            slices_ini = df.loc[:,u].unique()
            median = df.loc[:,u].unique()[0]

        # get center data
        temp_list = []

        tip_names = [s+'_c' for s in tip]
        col_names = ['v'] + tip_names 
        if not probe=='ftir':
            col_names = col_names + self.probe_settings[probe]['names'][h:]
        col_name_ordered = ['v'] + tip_names

        for col in [v] + self.probe_settings[probe]['names'][2:]:
            temp_list.append(df.loc[df.loc[:,u] == median, col].reset_index(drop = True))

        if slices_ini.size>=3:
            for p in tip:
                for k in slices_ini[[0,-1]]:    
                    temp_list.append(df.loc[df.loc[:,u] == k,p].reset_index(drop = True))
                col_names = col_names + [p + '_l', p + '_r']
                col_name_ordered = col_name_ordered + [p + '_l', p + '_r']

        df_out = pd.concat(temp_list, axis = 1, ignore_index=True)

        df_out.columns =  col_names

        # re order slices
        if not probe=='ftir':
            col_name_ordered = col_name_ordered + self.probe_settings[probe]['names'][h:]
        df_out = df_out.loc[:, col_name_ordered]

        # add extra information (in case needed for statistical analysis)
        for x in self.info_columns:
            df_out[self.info_columns_short[x]] = run_info[x]

        return df_out
    
    def _subset_info(self, df, **kwargs):
        df_probe = df.copy()
        key_list = []
        for key in kwargs.keys():
            key_list.append(key)
            df_probe = df_probe.loc[(df[key]==kwargs[key])]
        df_probe = df_probe.drop(columns = key_list).copy()
        df_probe.fillna('', inplace = True)
        df_probe.reset_index(inplace = True, drop = True)

        return df_probe
    
    def _angle_var_int(self, x):
        try:
            x = np.int(x)
        except:
            x = int(0)
        
        return x
    
    def read(self, datapath = './', probe = None, only_data_cols = True, **kwargs):
        read_error = ''
        try:
            df = self.read_data(datapath, probe = probe, only_data_cols = only_data_cols, **kwargs)
            flag = True
        except Exception as e:
            warnings.warn('\ncould not load %s. \traceback self.read_data:\n%s' % (datapath, e))
            df = self._empty_frame
            flag = False
            
        if np.logical_and(flag, probe == 'vels'):
            # print('fixing vels')
            df.loc[:, 'angle'] = df.loc[:, 'angle'].apply(self._angle_var_int)
            df = df.pivot_table(index = ['x', 'y'], columns = ['angle'], values = ['vp', 'vs'], aggfunc = self.vel_agg).reset_index(drop = False)
            mi = df.columns.to_list()
            df.columns = pd.Index([str(e[0]) + str(e[1]) for e in mi])
        return df
    
    def read_sample_data(self, fdesc, datapath = './', **kwargs):
        r  = fdesc.index.unique().values[0]
        s = len(fdesc)
        try:
            df_gen = (self.read(datapath = os.path.join(datapath, v.relroot), 
                                probe = v.probe, 
                                **kwargs) 
                    for _,v in fdesc.iterrows())        
            flag = True
        except Exception as e:
            warnings.warn('traceback error: %s' %(e))
            flag = False
            df_gen = (self._empty_frame)
            df = self._empty_frame
            
        
        if flag:
            if s > 1:
                df = reduce(lambda left,right: pd.merge(left, right, on = ['x', 'y'], how ='outer'), df_gen)
            elif s == 1:
                df = [*df_gen][0]
        
            df = pd.merge(left = self._empty_frame, right = df, how = 'outer')#.loc[:, self._empty_frame.columns]
            df = pd.concat([df], keys = [r], names = self.sample_data_index_names)
        else:
            warnings.warn('no data to loaded. %s is empty.' % (fdesc))
            df = None
        
        return df
    
    def __init__(self, labutilspath=None):
        super().__init__(labutilspath=labutilspath)
        self._reduce = reduce
        self.sample_data_index_names = ['family', 'code', 'tag', 'subtag', 'instance', 'experiment', 'side', 'm']
        self._empty_frame = self.empty_dataframe
        return
