import os
import git

def generate_config(dataset, args, exp_name) :
    
    repo = git.Repo(search_parent_directories=True)

    if args.encoder == 'lstm' :
        enc_type = 'rnn'
    elif args.encoder == 'average' :
        enc_type = args.encoder
    else :
        raise Exception("unknown encoder type")

    config = {
        'model' :{
            'encoder' : {
                'vocab_size' : dataset.vec.vocab_size,
                'embed_size' : dataset.vec.word_dim,
		'type' : enc_type,
		'hidden_size' : args.hidden_size
            },
            'decoder' : {
                'attention' : {
                    'type' : 'tanh'
                },
                'output_size' : dataset.output_size
            }
        },
        'training' : {
            'bsize' : dataset.bsize if hasattr(dataset, 'bsize') else 32,
            'weight_decay' : 1e-5,
            'pos_weight' : dataset.pos_weight if hasattr(dataset, 'pos_weight') else None,
            'basepath' : dataset.basepath if hasattr(dataset, 'basepath') else 'outputs',
            'exp_dirname' : os.path.join(dataset.name, exp_name)
        },
        'git_info' : {
            'branch' : repo.active_branch.name,
            'sha' : repo.head.object.hexsha
        },
        'command' : args.command
    }

    if args.encoder == 'average' :
    	config['model']['encoder'].update({'projection' : True, 'activation' : 'tanh'})

    return config

