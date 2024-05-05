import pyglet
import pymunk
from SyMBac.cell import Cell
from copy import deepcopy
import pickle
from scipy.stats import norm

def create_space():
    space = pymunk.Space(threaded=True)
    space.threads = 2
    return space

def update_cell_lengths(cells):
    for cell in cells:
        cell.update_length()


def update_pm_cells(cells):
    for cell in cells:
        if cell.is_dividing():
            daughter_details = cell.create_pm_cell()
            if len(daughter_details) > 2: # Really hacky. Needs fixing because sometimes this returns cell_body, cell shape. So this is a check to ensure that it's returing daughter_x, y and angle
                daughter = Cell(**daughter_details)
                cell.daughter = daughter
                cells.append(daughter)
        else:
            cell.create_pm_cell()

def update_cell_positions(cells):
    for cell in cells:
        cell.update_position()

def wipe_space(space):
    for body, poly in zip(space.bodies, space.shapes):
        if body.body_type == 0:
            space.remove(body)
            space.remove(poly)        



def step_and_update(dt, cells, space, phys_iters, ylim, cell_timeseries,x,sim_length,save_dir):
    for shape in space.shapes:
        if shape.body.position.y < 0 or shape.body.position.y > ylim:
            space.remove(shape.body, shape)
    #new_cells = []
    graveyard = []
    for cell in cells:
        if cell.shape.body.position.y < 0 or cell.shape.body.position.y > ylim:
            graveyard.append([cell, "outside"])
            cells.remove(cell)
        elif norm.rvs() <= norm.ppf(cell.lysis_p) and len(cells) > 1:   # in case all cells disappear
            graveyard.append([cell, "lysis"])
            cells.remove(cell)
        else:
            pass
            #new_cells.append(cell)
    #cells = deepcopy(new_cells)
    graveyard = deepcopy(graveyard)

    wipe_space(space)

    update_cell_lengths(cells)
    update_pm_cells(cells)

    for _ in range(phys_iters):
        space.step(dt)
    update_cell_positions(cells)

    #print(str(len(cells))+" cells")
    if x[0] > 1:
        cell_timeseries.append(deepcopy(cells))
    if x[0] == sim_length-1:
        with open(save_dir+"/cell_timeseries.p", "wb") as f:
            pickle.dump(cell_timeseries, f)
        with open(save_dir+"/space_timeseries.p", "wb") as f:
            pickle.dump(space, f)
        pyglet.app.exit()
        return cells
    x[0] += 1
    return (cells)


