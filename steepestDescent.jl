using Plots
using ForwardDiff
using LinearAlgebra

plotly()

Plots.PlotlyBackend()



# Function that computes the value of the objective function
function objFunctContour(x)
    functVal =
        1 -
        (1 + cos(12 * sqrt(x[1]^2 + x[2]^2))) / (0.5 * (x[1]^2 + x[2]^2) + 2)
    return functVal
end


function objFunct(x,y)
    functVal =
        1 -
        (1 + cos(12 * sqrt(x^2 + y^2))) / (0.5 * (x^2 + y^2) + 2)
    return functVal
end


function SteepDesKernel(xin)
    #initial calculations
    x = xin
    Xvals=[]
    Yvals=[]
    c = objFunctContour(x)
    icount = 1
    iter = 0
    g = ForwardDiff.gradient(objFunctContour, x)
    gradMagCheck=norm(g)
    gradMag=gradMagCheck
    # steepest descent optimization
    count=0
    while gradMag >= 1e-9
        iter = iter + 1
        count = count + 1
        c = objFunctContour(x)
        g = ForwardDiff.gradient(objFunctContour, x)


        atry=1
        # line search - backtracking
        flag=true
        while flag== true
            xtry = x - atry * g
            ctry = objFunctContour(xtry)
            icount = icount + 1
            if (ctry <= (10^(-4))*atry*transpose(g)*(-g) +c)
                x = xtry
                c = ctry
                flag=false
            end
            atry=atry*.75
            println("grD: ",atry)
        end

        if count>= 100
            count=0
            g2=norm(g)
            diffs=gradMagCheck-g2
            checks=diffs/gradMagCheck
            if checks<0.05
                xmin = x
                cmin = c
                return x,c,Xvals,Yvals
            end
            gradMagCheck=g2
        end
        gradMag=norm(g)

        push!(Xvals, x[1])
        push!(Yvals, x[2])

    end

    # minimum





end

xin=[2,-2]

x,c,valsX, valsY =SteepDesKernel(xin)

x = -2:0.05:2
y = -2:0.05:2
f(x, y) = begin
    objFunct(x, y)
end
#X = repeat(reshape(x, 1, :), length(y), 1)
#Y = repeat(y, 1, length(x))
#Z = map(f, X, Y)
contour(x, y, f)
scatter!(valsX, valsY, label = "Some points")
