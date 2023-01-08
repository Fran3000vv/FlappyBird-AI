import os
import random
import time
import pygame
import math as ma
import neat
import pickle

WIN_WIDTH=500
WIN_HEIGHT=700
BIRD_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join("img","bird1.png")))
BG_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join("img","bg.png")))
PIPE_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join("img","pipe.png")))
pygame.font.init()#Iniciar las Fuentes pygame
class Bird():
    IMG=BIRD_IMG
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.tilt=0
        self.tick_count=0
        self.vel=0
        self.img=self.IMG
    def jump(self):
        self.tick_count=0
        self.vel=-20
    def draw(self,window):
        rotated_img=pygame.transform.rotate(self.img,self.tilt)
        new_rectangle=rotated_img.get_rect(center=self.img.get_rect(topleft=(self.x,self.y)).center)
        window.blit(rotated_img,new_rectangle.topleft)
    def move(self,points):
        self.tick_count+=1
        gravity=9.8
        d=self.vel*self.tick_count+0.5*gravity*(self.tick_count**2)
        '''La verdad que me he puesto a hacer una función cualquiera en el desmos
        hasta que satisfacera lo que buscaba, que es que hasta antes del 40-50 creciera
        menos que el propio valor y a partir de ese intervalo más, porque las bajadas deben
        ser más rápidas en altas velocidades(se que no es la más sencilla para lo que quería).
        Iba a poner un abs(), pero me parecio bien que al principio fuera menor
        La verdad podría ser diferente, pero la primera vez que hice un pequeño esbozo me
        gusto y quise conserar su forma aunque fuera rara la función
        Opciones Parecidas serían un (x*0.15)^2 o (0.08*x)^3, pero que esa me gusta más
        La verdad es que es bastante exponencial, pero para IA quiza works well enough'''
        if d>25+points*1.5*ma.log((points+1)*0.1):
            d=25
        self.y=self.y+d
class Pipe():
    IMG_UP=PIPE_IMG
    IMG_DOWN=pygame.transform.flip(PIPE_IMG,False,True)
    GAP=400
    def __init__(self):
        self.x=WIN_WIDTH
        self.height=random.randint(50,WIN_HEIGHT-50-self.GAP)
        self.top=self.height-self.IMG_UP.get_height()
    def draw(self,window):
        window.blit(self.IMG_DOWN,(self.x,self.top))
        window.blit(self.IMG_UP,(self.x,self.height+self.GAP))
    def move(self,points):
        self.x-=5+points*0.5#Arreglo para la velocidad in crescendo
    #Función Para comprobar Colisión viendo principio y final de alto y ancho si están en territorio de una tubería
    def col(self,b):
        return ((self.x<=b.x<=self.x+self.IMG_UP.get_width()) or (self.x<=b.x+b.IMG.get_width()<=self.x+self.IMG_UP.get_width())) and (not (self.height<=b.y<=self.height+self.GAP) or not (self.height<=b.y+b.IMG.get_height()<=self.height+self.GAP))



def draw_window(win,bird,pipe,points):
    win.blit(BG_IMG,(0,0))
    bird.draw(win)
    pipe.draw(win)
    #Pintar Texto de la Puntuación
    font = pygame.font.SysFont("Carlito Bold", 70)
    img_txt = font.render(str(points),True,(0,0,0))
    win.blit(img_txt,(WIN_WIDTH/2,50))
    pygame.display.update()


def game():
    bird=Bird(200,300)
    window=pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    pipe=Pipe()
    points=0
    run=True
    clock=pygame.time.Clock()
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            #Detectar todos los eventos que se producen
            #Cuando se quita el Juego
            if event.type==pygame.QUIT:
                run=False
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_SPACE:
                    bird.jump()
        bird.move(points)
        pipe.move(points)
        run=not pipe.col(bird) #Establece el bucle del while true en función de si el pájaro ha colisionado con las tuberías o no
        #Suma Puntos
        if(pipe.x<bird.x+bird.IMG.get_width() and pipe.x+5+points*0.5>=bird.x+bird.IMG.get_width()):#Al Principio porque suele ser al principio de la tubería y no al final o en el medio
            points+=1
        if pipe.x+pipe.IMG_UP.get_width()<0:
            pipe=Pipe()
        draw_window(window,bird,pipe,points)#Le paso los puntos para visualizarlos y para el incremento
    print("Has Tenido Una Puntuación de",points)#Cuando le pierdo Printeo puntos
#game()
def test_ai(gen):
    net=gen
    bird=Bird(200,300)
    window=pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    pipe=Pipe()
    points=0
    run=True
    clock=pygame.time.Clock()
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                quit()
        output=net.activate((bird.y,pipe.height,pipe.x-bird.x))
        decision=output.index(max(output))
        if decision==0:
            pass
        elif decision==1:
            bird.jump()
        bird.move(points)
        pipe.move(points)
        if pipe.col(bird) or bird.y>WIN_HEIGHT:
            quit()
        if(pipe.x<bird.x+bird.IMG.get_width() and pipe.x+5+points*0.5>=bird.x+bird.IMG.get_width()):
            points+=1
        if pipe.x+pipe.IMG_UP.get_width()<0:
            pipe=Pipe()
        draw_window(window,bird,pipe,points)
def train_ai(gen,config):
    net=neat.nn.FeedForwardNetwork.create(gen,config)
    bird=Bird(200,300)
    window=pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    pipe=Pipe()
    points=0
    run=True
    times=time.time()
    while run:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                quit()
        output=net.activate((bird.y,pipe.height,pipe.x-bird.x))
        decision=output.index(max(output))
        if decision==0:
            pass
        elif decision==1:
            bird.jump()
        bird.move(points)
        pipe.move(points)
        if pipe.col(bird) or bird.y>WIN_HEIGHT:
            break
        if(pipe.x<bird.x+bird.IMG.get_width() and pipe.x+5+points*0.5>=bird.x+bird.IMG.get_width()):
            points+=1
        if pipe.x+pipe.IMG_UP.get_width()<0:
            pipe=Pipe()
        draw_window(window,bird,pipe,points)
    gen.fitness=points+time.time()-times
def eval_genomes(genomes,config):
    for i,(gen_id,gen) in enumerate(genomes):
        gen.fitness=0
        train_ai(gen,config)
def test_best_ai(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net=neat.nn.FeedForwardNetwork.create(winner,config)
    test_ai(winner_net)
def run_neat(config):
    p=neat.Checkpointer.restore_checkpoint('neat-checkpoint-209')
    #p=neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(1))
    winner=p.run(eval_genomes,1)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
    print(winner)

if __name__=="__main__":
    config_path=os.path.join(os.path.dirname(__file__),"config.txt")
    config=neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)
    #run_neat(config)
    test_best_ai(config)