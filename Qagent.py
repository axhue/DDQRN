
actions = {
    'strafe':{
        'left': 'strafe -1',
        'right': 'strafe 1'
    },
    'move':{
        'back':'move -1',
        'forward':'move 1'
    }
}

# Q needs a function with 2 input s,a and output 1 expected reward
class QAgent:
