
import os, sys, re, shutil, warnings, string, fnmatch
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
            'names':['x','y','perm','meas_code','tile'],
            'tip':['perm'],
            'h':3
        },
        'impulse':{
            'usecols':[0,1,2,3],
            'skiprows':7,
            'names':['x','y','e_star','tile'],
            'tip':['e_star'],
            'h':3
        },
        'vel':{
            'usecols':[0,1,2,3,6,9],
            'skiprows':7,
            'names':['x','y','angle','vp','vs','tile'],
            'tip':['vp','vs'],
            'h':5
        },
        'ftir':{
            'usecols':None,
            'skiprows':2,
            'names':['x','y']+['l_'+str(int(x)) for x in np.linspace(1,1752,1752)],
            'tip':['l_'+str(int(x)) for x in np.linspace(1,1752,1752)],
            'h':None
        }
        
    }

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
        df = df.apply(pd.to_numeric,errors='coerce').dropna()
        df.replace(np.inf, np.nan, inplace = True)
        df.dropna(inplace = True)
        # reset index 
        df.reset_index(inplace = True, drop = True)

        return df

    def read_data(self, fpath, probe, zero_offset = True, fix_xy = True):
        df = pd.read_csv(fpath, 
                        usecols = self.probe_settings[probe]['usecols'],
                        skiprows = self.probe_settings[probe]['skiprows'],
                        names = self.probe_settings[probe]['names'])

        df = self._enforce_float(df)

        if fix_xy:
            df.loc[:,['x','y']] = df.loc[:,['x','y']].apply(self._fix_xy)
        # reset x,y offset to zero
        if zero_offset:
            df.iloc[:,:2] = df.iloc[:,:2] - df.iloc[:,:2].min()
        return df
    
    def _save(self, data, method= 'pandas', savefile = 'data'):
        if method =='pandas':
            data.to_csv(savefile + '.csv', index = False)
        if method=='h5' or method=='deepdish':
            dp.io.save(savefile + '.h5', data)
        return
    
    def save_data(self, df, method = 'pandas', savepath = './', savename = 'data', save_xyp = False, probe = None):
        savefile = os.path.join(savepath, savename)
        if save_xyp and (probe is not None):
            h = self.probe_settings[probe]['h']
            df = df.iloc[:,:h].copy()
        self._save(df, savefile = savefile, method = method)
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

    _df_colnames_ordered = ['sample_tag', 'subsample_tag', 'sample_code', 'sample_family', 'probe', 'side', 'instance', 'fname', 'relroot']

    def _get_exclude_list(self):
        self.datapath['exclude'] = ['_special-studies', 'special_studies', '_special_studies', '_unsorted', 
        self.datapath['raw'], self.datapath['generic'], self.datapath['analysis'], '*layout*', '_postprocessed']
        self._files_excludes = r'|'.join([fnmatch.translate(x) for x in self.datapath['files']['exclude']]) or r'$.'
        self._files_includes = r'|'.join([fnmatch.translate(x) for x in self.datapath['files']['include']]) or r'$.'
        self._datapath_length = int(len(self.datapath['path'].split('/')))
        return
    
    def _rename_file(self, root,fname_old, fname_new):
        oldname = os.path.join(root,fname_old)
        newname = os.path.join(root,fname_new)
        shutil.move(oldname,newname)
        return

    def _get_refindall(self, pattern, string):
        v = None
        s = re.findall(pattern, string)
        if len(s)>0:
            v = s[0]
        return v

    def _get_sides(self, x):
        side = self._get_refindall(r'(before|after)_([a-z]+|[0-9]+)[.]', x)
        if side is not None:
            side = side[-1]
        return side

    def _get_subsample(self, x):
        sub = self._get_refindall(r'.*sub[a-z]+[/]([a-z]+|[0-9]+|[a-z]+[0-9]+|[a-z]+[0-9]+[_]+[a-z]+|[a-z]+[0-9]+[_]+[a-z]+[0-9]+)[/]',x)
        return sub

    def _get_probename(self, x):
        probe = self._get_refindall(r'(perm|vel|impulse|ftir)',x)
        return probe

    def _get_instance(self, x):
        instance = self._get_refindall(r'(before|after)', x)
        return instance

    def add_before_fname(self, fname,root):
        name2 = re.sub(r'(perm|vel|impulse|ftir).*([a-z]+)[.]([a-z]+)',r'\1_before_\2.\3',fname)
        if self.debug:
            print(root.split('/')[self._datapath_length - 1], fname, name2, sep='\t')
        if not self.dryrun:
            self._rename_file(root, fname, name2)
        return name2

    def swap_instance_fname(self, fname,root):
        name2 = re.sub(r'(perm|vel|impulse|ftir)(_|-)([a-z]+|[0-9]+)(_|-).*(before|after).*[.]([a-z]+)',
                    r'\1_\5_\3.\6',
                    fname)
        if self.debug:
            print('swaping', root.split('/')[self._datapath_length - 1], fname,name2,sep='\t')
        if not self.dryrun:
            self._rename_file(root, fname, name2)
        return name2

    def check_autoscan_fname(self, fname, root):
        instance = None
        if (not 'before' in fname) and (not 'after' in fname):
            warnings.warn('before or after not found in ' + 
                        root.split('/')[self._datapath_length - 1] + fname)
            fname = self.add_before_fname(fname, root)
        instance = re.findall('before|after',fname)
        if len(instance)==1:
            instance = instance[0]
            if self.debug: print(root.split('/')[self._datapath_length - 1], instance, fname, sep='\t')
            if len(fname.split('_'))>2:
                tst = re.match(r'(perm|vel|impulse|ftir)(-|_)(before|after).*([a-z]+)[.]([a-z]+)', fname)
                if tst is None:
                    fname = self.swap_instance_fname(fname, root)
        else:
            if len(instance)>1:
                warning = fname + ' has more than one instance: ' + ', '.join(instance)
                warning = warning + ' and cannot choose! \n check ' + os.path.join(root,fname) 
            if len(instance)==0:
                warning = fname + ' not in either category (!) \n please review!'
            warnings.warn(warning)
        return fname
    
    def wrangle(self, save = False, link = False, savepath = None):
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
                rs.append(os.path.relpath(os.path.join(root,fname),start=self.datapath['path']))

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
            df['sample_' + s] = df['sample_tag'].apply(self._get_rockinfo, key=s)

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
    
    shortnames_dict = {
        'side':'side',
        'sample_code':'code',
        'sample_family':'family',
        'sample_tag':'tag',
        'subsample_tag':'subtag'
        }
    
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
                varout = self.shortnames_dict[var]
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
            data[self.shortnames_dict[x]] = run_info[x]

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
        col_names = []

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
            df_out[self.shortnames_dict[x]] = run_info[x]

        return df_out
    
    def _subset_info(self, df, **kwargs):
        df_probe = df.copy()
        key_list = []
        for key in kwargs.keys():
            key_list.append(key)
            df_probe = df_probe.loc[(df[key]==kwargs[key])]
        df_probe = df_probe.drop(columns = key_list).copy()
        df_probe.fillna('', inplace=True)
        df_probe.reset_index(inplace=True, drop=True)

        return df_probe
    
    def __init__(self, labutilspath=None):
        super().__init__(labutilspath=labutilspath)
        return
