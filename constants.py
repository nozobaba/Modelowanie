# Stałe i parametry symulacji

DT = 30.0       # krok czasowy [s]
HX = 0.2        # komórka siatki [m]

ALPHA = 1.0e-4  # dyfuzja termiczna [m^2/s]

P_RADIATOR_W = 2000.0   # moc grzejnika [W]
RHO_AIR = 1.293          # gęstość powietrza [kg/m^3]
C_AIR = 1005.0            # ciepło właściwe [J/(kg*K)]
CEILING_HEIGHT_M = 2.5    # wysokość sufitu [m]
WALL_LOSS_BETA = 0.08     # straty przez ściany zew. (0..1)

APARTMENT_WIDTH_M = 10.0
APARTMENT_HEIGHT_M = 8.0

SETPOINT_HOME_C = 21.0    # komfort [°C]
SETPOINT_AWAY_C = 15.0    # oszczędna [°C]
