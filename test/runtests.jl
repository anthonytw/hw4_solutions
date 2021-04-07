using Test
using NBInclude

@testset "HW4" begin
    @testset "Question 1" begin
        println("Testing Question 1...")
        @nbinclude(joinpath(@__DIR__, "..", "src", "Q1.ipynb"))
    end
    @testset "Question 2" begin
        using NBInclude
        println("Testing Question 2...")
        @nbinclude(joinpath(@__DIR__, "..", "src", "Q2.ipynb"))
    end
    @testset "Question 3" begin
        println("Testing Question 3...")
        @nbinclude(joinpath(@__DIR__, "..", "src", "Q3.ipynb"))
    end
end