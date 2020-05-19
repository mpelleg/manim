from manimlib.imports import *
import os
import pyclbr
from pdb import set_trace

class PlotStepFow(GraphScene):
    CONFIG = {
        "x_min": 0,
        "x_max": 11,
        "x_axis_width": 9,
        "x_tick_frequency": 1,
        "x_leftmost_tick": None, # Change if different from x_min
        "x_labeled_nums": None,
        "x_axis_label": "$t[s]$",
        "y_min": 0,
        "y_max": 2,
        "y_axis_height": 6,
        "y_tick_frequency": 1,
        "y_bottom_tick": None, # Change if different from y_min
        "y_labeled_nums": None,
        "y_axis_label": "$F_{d}$",
        "axes_color": GREY,
        "graph_origin": 2.5 * DOWN + 4 * LEFT,
        "exclude_zero_label": True,
        "num_graph_anchor_points": 25,
        "default_graph_colors": [BLUE, GREEN, YELLOW],
        "default_derivative_color": GREEN,
        "default_input_color": YELLOW,
        "default_riemann_start_color": BLUE,
        "default_riemann_end_color": GREEN,
        "area_opacity": 0.8,
        "num_rects": 50,
        "function_color" : RED,
        "x_labeled_nums" :range(0,10,1),
        "y_labeled_nums" :range(0,1,1)

    }

    def construct(self):
        self.x1=0.98
        self.x2=1.02
        self.setup_axes(animate=True)
        func_graph=self.get_graph(self.func_to_graph,self.function_color)
        func_graph1=self.get_graph(self.func_to_graph1,self.function_color,0,self.x1)
        func_graph2=self.get_graph(self.func_to_graph2,self.function_color,self.x1,self.x2)
        func_graph3=self.get_graph(self.func_to_graph3,self.function_color,self.x2)

        #func_graph2=self.get_graph(self.func_to_graph2)
        vert_line = self.get_vertical_line_to_graph(TAU,func_graph,color=YELLOW)
        graph_lab = self.get_graph_label(func_graph, label = "Fd")
        #graph_lab2=self.get_graph_label(func_graph2,label = "\\sin(x)", x_val=-10, direction=UP/2)
        two_pi = TexMobject("x = 2 \\pi")
        label_coord = self.input_to_graph_point(TAU,func_graph)
        two_pi.next_to(label_coord,RIGHT+UP)



        #self.play(ShowCreation(func_graph))
        
        #self.play(ShowCreation(vert_line), ShowCreation(graph_lab))#,ShowCreation(two_pi))
        self.play(ShowCreation(func_graph1))
        self.play(ShowCreation(func_graph2))
        self.play(ShowCreation(func_graph3))


    
    def func_to_graph(self,x):
        y=x*0
        x1=0.98
        x2=1.02
        if x>=x1 and x<x2:
           y=(x-x1)*0.8/(abs(x2-x1));
        if x>=x2:
           y=x*0+0.8;

        return y

    def func_to_graph1(self,x):
        y=x*0
        return y
    
    def func_to_graph2(self,x):
        x1=0.98
        x2=1.02
        y=(x-x1)*0.8/(abs(x2-x1));
        return y
    def func_to_graph3(self,x):
        y=(x*0.)+0.8;
        return y


class GraphFromData(GraphScene):
    # Covert the data coords to the graph points
    def get_points_from_coords(self,coords):
        return [
            # Convert COORDS -> POINTS
            self.coords_to_point(px,py)
            # See manimlib/scene/graph_scene.py
            for px,py in coords
        ]

    # Return the dots of a set of points
    def get_dots_from_coords(self,coords,radius=0.1):
        points = self.get_points_from_coords(coords)
        dots = VGroup(*[
            Dot(radius=radius).move_to([px,py,pz])
            for px,py,pz in points
            ]
        )
        return dots


class PlotStepFow2(GraphFromData):
    CONFIG = {
        "x_min": 0,
        "x_max": 11,
        "x_axis_width": 9/2,
        "x_tick_frequency": 1,
        "x_leftmost_tick": None, # Change if different from x_min
        "x_labeled_nums": None,
        "x_axis_label": "$t[s]$",
        "y_min": 0,
        "y_max": 1.0001,
        "y_axis_height": 2,
        "y_tick_frequency": 0.5,
        "y_bottom_tick": None, # Change if different from y_min
        "y_labeled_nums": None,
        "y_axis_label": "$F_{d}$",
        "axes_color": GREY,
        "graph_origin": 3.0 * DOWN + 6 * LEFT,
        "exclude_zero_label": True,
        "num_graph_anchor_points": 25,
        "default_graph_colors": [BLUE, GREEN, YELLOW],
        "default_derivative_color": GREEN,
        "default_input_color": YELLOW,
        "default_riemann_start_color": BLUE,
        "default_riemann_end_color": GREEN,
        "area_opacity": 0.8,
        "num_rects": 50,
        "function_color" : RED,
        "x_labeled_nums" :range(0,11,1),
        "y_labeled_nums" :[0,0.5,1]

    }

    def construct(self):

        pump = SVGMobject("pump_copy")
        pump.set_fill(WHITE, opacity = 0)
        pump.circle=pump.submobjects[0]
        pump.triangle=pump.submobjects[4]
        pump.line1=pump.submobjects[1]
        pump.line2=pump.submobjects[2]
        pump.bottom=pump.submobjects[3]
        pump.pline = pump.submobjects[5]
        pump.cylinder = pump.submobjects[6]
        pump.head = pump.submobjects[7]
        pump.rod = pump.submobjects[8]



        pump.triangle.set_fill(pump.color, opacity = 1)
        pump.scale(1.5)
        pump.shift(1.5*UP)
        
        arrow = Arrow(ORIGIN+LEFT*3.3+UP*0.3,ORIGIN+UP*2.7+LEFT*2.6)

        self.play(FadeIn(pump),FadeIn(arrow))


        self.setup_axes(animate=True)

        coords = [[0,0],[1,0],[1,0.8],[2,0.8],[3,0.8],[4,0.8],[5,0.8],[6,0.8],[7,0.8],[8,0.8],[9,0.8],[10,0.8]]
        points = self.get_points_from_coords(coords)
        # Set graph
        graph = DiscreteGraphFromSetPoints(points,color=ORANGE)
        func_graph=self.get_graph(self.func_to_graph,self.function_color)
        # Set dots
        dots = self.get_dots_from_coords(coords)
        self.play(ShowCreation(graph,run_time=1.5),Rotate(arrow, -PI/4))
        self.wait(2)  
        f1 = TextMobject("{\\Huge $\\Rightarrow$}")
        f1.scale(0.7)
        label_coord1 = self.input_to_graph_point(10,func_graph)
        f1.next_to(label_coord1,6*RIGHT+3*DOWN)
        self.play(FadeIn(f1))


        self.graph_origin = 3 * DOWN + 2* RIGHT
        self.y_axis_label = "$p$"
        self.y_max = 200
        self.y_tick_frequency=50
        self.y_labeled_nums = range(0,200,50)
        self.setup_axes(animate=True)
        func_graph2=self.get_graph(self.func_to_graph_bessel,self.function_color,1,10)
        #func_graph3=self.get_inv_graph(lambda y: np.sin(0.05*y),y_min=0,y_max=150)
        #path = self.get_inv_graph(lambda x: ((100*(x-2.5))**80)*100+363+5*np.sin(500*(x-2.5)), x_min=2.5, x_max=2.51)
        #path = self.get_inv_graph(lambda y: ((100*(y-2.5))**80)*100+363+5*np.sin(500*(y-2.5)), y_min=2.5, y_max=2.51)
        v = np.arange(100)/10
        vv = (v+0.3*np.sin(3*v))*15+363
        vv2 = (v+0.3*np.sin(3*v))*15+473
        z = np.ones((100))*2.51
        coords2 = np.stack((z, vv), axis=-1).tolist()
        coords3 = np.stack((z, vv2), axis=-1).tolist()

        points2 = self.get_points_from_coords(coords2)
        points3 = self.get_points_from_coords(coords3)

        path = DiscreteGraphFromSetPoints(points2,color=ORANGE)
        path2 = DiscreteGraphFromSetPoints(points3,color=ORANGE)

        location = self.coords_to_point(2.5,363) #location: Point
        self.play(MoveAlongPath(pump.head,path,run_time=5),MoveAlongPath(pump.rod,path2,run_time=5),ShowCreation(func_graph2,run_time=5))
        #self.play(MoveAlongPath(pump.head, path,run_time=2))

        #self.play(ShowCreation(func_graph2,run_time=2))




    def get_points_from_coords(self,coords):
        return [
            # Convert COORDS -> POINTS
            self.coords_to_point(px,py)
            # See manimlib/scene/graph_scene.py
            for px,py in coords
        ]

    def func_to_graph(self,x):
        y=x*0
        x1=0.98
        x2=1.02
        if x>=x1 and x<x2:
           y=(x-x1)*0.8/(abs(x2-x1));
        if x>=x2:
           y=x*0+0.8;

        return y
    def func_to_graph_bessel(self,x):
        return -150*np.sin(3*(x-0.9))/(3*(x-0.9))+150

        return y

    def get_inv_graph(
        self, func,
        color=None,
        y_min=None,
        y_max=None,
        **kwargs
    ):
        if color is None:
            color = next(self.default_graph_colors_cycle)
        if y_min is None:
            y_min = self.x_min
        if y_max is None:
            y_max = self.x_max

        def parameterized_function(alpha):
            y = interpolate(y_min, y_max, alpha)
            x = func(y)
            if not np.isfinite(x):
                x = self.x_max
            return self.coords_to_point(x, y)

        graph = ParametricFunction(
            parameterized_function,
            color=color,
            **kwargs
        )
        graph.underlying_function = func
        return graph

class DiscreteGraphFromSetPoints(VMobject):
    def __init__(self,set_of_points,**kwargs):
        super().__init__(**kwargs)
        self.set_points_as_corners(set_of_points)
class SmoothGraphFromSetPoints(VMobject):
    def __init__(self,set_of_points,**kwargs):
        super().__init__(**kwargs)
        self.set_points_smoothly(set_of_points)





if __name__ == "__main__":
    # Call this file at command line to make sure all scenes work with version of manim
    # type "python manim_tutorial_P37.py" at command line to run all scenes in this file
    #Must have "import os" and  "import pyclbr" at start of file to use this
    ###Using Python class browser to determine which classes are defined in this file
    module_name = 'matteo_manim'   #Name of current file
    module_info = pyclbr.readmodule(module_name)

    for item in module_info.values():
        if item.module==module_name:
            print(item.name)
            os.system("python -m manim matteo_manim.py %s -l" % item.name)  #Does not play files

