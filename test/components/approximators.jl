@testset "Approximators" begin

    @testset "TabularApproximator" begin
        A = TabularApproximator(ones(3))

        @test A(1) == 0.0
        @test A(2) == 0.0

        update!(A, 2 => 3.0)
        @test A(2) == 3.0
    end

end
