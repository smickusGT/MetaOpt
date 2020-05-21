
function CostFunction(x)
    functVal =
        1 -
        (1 + cos(12 * sqrt(x[1]^2 + x[2]^2))) / (0.5 * (x[1]^2 + x[2]^2) + 2)
    return functVal
end
# Particle Swarm Optimization Kernel

function PsoKernel(xin, vin, w, c1, c2, npart, niter)
    #npart is the number of particles
    #xin the intial conditions?

    # initial calculations
    x = xin
    v = vin
    c = zeros(1, npart)

    for i = 1:npart
        c[i] = CostFunction(x[:, i])
    end

    xpbest = x
    vpbest = v
    cpbest = c

    cgbest  = minimum(c)
    ibest = argmin(c)
    xgbest = x[:, ibest[2]]
    vgbest = v[:, ibest[2]]

    #pso iterations

    for i = 1:niter

        #pso velocity and position

        r1 = rand(Float64,1)
        r2 = rand(Float64,1)
        v =
        w * v +
            (c1 * r1) .* (xpbest - x) +
            (c2 * r2) .* (repeat(xgbest, 1, npart) - x)
        x = x + v

        # compute cost for updated particles

        for i = 1:npart
            c[i] = CostFunction(x[:, i])
        end

        # update individual and global particles

        for i = 1:npart
            if (c[i] < cpbest[i])
                cpbest[i] = c[i]
                xpbest[:, i] = x[:, i]
                vpbest[:, i] = v[:, i]
            end
            if (c[i] < cgbest)
                cgbest = c[i]
                xgbest = x[:, i]
                vgbest = v[:, i]
            end
        end

        return x, v, c, xpbest, vpbest, cpbest, xgbest, vgbest, cgbest
    end
end

function InitialPartSwarmPoint(NumVars,Numpoints,MaxValVars,MaxValVarsVel)
    x=MaxValVars.*rand(Float64,NumVars,Numpoints)
    vel=MaxValVarsVel.*rand(Float64,NumVars,Numpoints)
    return x, vel
end

xin = [2 -1; 2 1]
vin = [10 5 ;10 5] #length needs to be equal to number of partilces also true for xin
xin, vin = InitialPartSwarmPoint(2,1000,4,10)
w = .5
c1 = 0.3
c2 = 5
npart = 1000
niter = 1000000

x, v, c, xpbest, vpbest, cpbest, xgbest, vgbest, cgbest =
    PsoKernel(xin, vin, w, c1, c2, npart, niter)
