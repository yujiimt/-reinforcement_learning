import os
import tornado.web
import tornado.escape
from chapter1_1 import Enviroment
from planner import ValuteIterationPlanner, PolicyIterationPlanner

class IndexHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("index.html")

class PlanningHandler(tornado.web.RequestHandler):

    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        grid = data["grid"]
        paln_type = data["plan"]
        move_prob = 0.8 # default value

        try:
            move_prob = float(data["prob"])
        except ValueError:
            pass

        env = Enviroment(grid, move_prob=move_prob)
        if paln_type == "value":
            planner = ValuteIterationPlanner(env)
        elif paln_type == "policy":
            planner = PolicyIterationPlanner(env)
                      
        
        result = planner.plan()
        planner.log.append(result)
        self.write({"log" : planner.log})

class Application(tornado.web.Application):

    def __init__(self):
        handlers = [
            (r"/", IndexHandler),
            (r"/", PlanningHandler),
        ]

        settings = dict(
            template_path = os.path.join(os.path.dirname(__file__), "templates"),
            static_path = os.path.join(os.path.dirname(__file__), "static"),
            cookie_secret = os.path.environ.get("SECRET_TOKEN", "__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE_"),
            debug = True,
        )

        super(Application, self).__init__(handlers=handlers, **settings)