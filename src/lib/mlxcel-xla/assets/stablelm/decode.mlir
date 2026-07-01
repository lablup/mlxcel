module @decode_step {
  func.func public @main(%arg0: tensor<32x32xf32> loc("params['embed']"), %arg1: tensor<32xf32> loc("params['final_norm']"), %arg2: tensor<32xf32> loc("params['final_norm_bias']"), %arg3: tensor<32x32xf32> loc("params['lm_head']"), %arg4: tensor<32x64xf32> loc("params['layers'][0]['down']"), %arg5: tensor<64x32xf32> loc("params['layers'][0]['gate']"), %arg6: tensor<32xf32> loc("params['layers'][0]['in_ln']"), %arg7: tensor<32xf32> loc("params['layers'][0]['post_ln']"), %arg8: tensor<64x32xf32> loc("params['layers'][0]['up']"), %arg9: tensor<16x32xf32> loc("params['layers'][0]['wk']"), %arg10: tensor<32x32xf32> loc("params['layers'][0]['wo']"), %arg11: tensor<32x32xf32> loc("params['layers'][0]['wq']"), %arg12: tensor<16x32xf32> loc("params['layers'][0]['wv']"), %arg13: tensor<16xf32> loc("params['layers'][0]['bk']"), %arg14: tensor<32xf32> loc("params['layers'][0]['bq']"), %arg15: tensor<16xf32> loc("params['layers'][0]['bv']"), %arg16: tensor<32xf32> loc("params['layers'][0]['in_ln_bias']"), %arg17: tensor<32xf32> loc("params['layers'][0]['post_ln_bias']"), %arg18: tensor<32x64xf32> loc("params['layers'][1]['down']"), %arg19: tensor<64x32xf32> loc("params['layers'][1]['gate']"), %arg20: tensor<32xf32> loc("params['layers'][1]['in_ln']"), %arg21: tensor<32xf32> loc("params['layers'][1]['post_ln']"), %arg22: tensor<64x32xf32> loc("params['layers'][1]['up']"), %arg23: tensor<16x32xf32> loc("params['layers'][1]['wk']"), %arg24: tensor<32x32xf32> loc("params['layers'][1]['wo']"), %arg25: tensor<32x32xf32> loc("params['layers'][1]['wq']"), %arg26: tensor<16x32xf32> loc("params['layers'][1]['wv']"), %arg27: tensor<16xf32> loc("params['layers'][1]['bk']"), %arg28: tensor<32xf32> loc("params['layers'][1]['bq']"), %arg29: tensor<16xf32> loc("params['layers'][1]['bv']"), %arg30: tensor<32xf32> loc("params['layers'][1]['in_ln_bias']"), %arg31: tensor<32xf32> loc("params['layers'][1]['post_ln_bias']"), %arg32: tensor<i32> loc("token"), %arg33: tensor<i32> loc("pos"), %arg34: tensor<i32> loc("cache_len"), %arg35: tensor<2x256x2x8xf32> loc("kcache"), %arg36: tensor<2x256x2x8xf32> loc("vcache")) -> (tensor<32xf32>, tensor<2x256x2x8xf32>, tensor<2x256x2x8xf32>) {
    %0 = stablehlo.constant dense<"0x0000803F0000803F40510A3F40510A3F3311D5BE3311D5BE26707DBF26707DBF305527BF305527BF2C3C913E2C3C913EB8CD753FB8CD753FBDFF403FBDFF403FF6FD14BEF6FD14BED53F69BFD53F69BF64CD56BF64CD56BF7205913B7205913BD006583FD006583F6F4E683F6F4E683FD7040C3ED7040C3EE87A42BFE87A42BF2C2975BF2C2975BF36E28CBE36E28CBE840A293F840A293FBF1B7D3FBF1B7D3F22F0D03E22F0D03EFC370CBFFC370CBF6FFD7FBF6FFD7FBFBF6708BFBF6708BFFE2DD93EFE2DD93E78BF7D3F78BF7D3F819C253F819C253F389395BE389395BE576D76BF576D76BFB3803FBFB3803FBF18F41D3E18F41D3E8E2C6A3F8E2C6A3FAA8F553FAA8F553FB78659BCB78659BCE73B59BFE73B59BF5F5867BF5F5867BFEA0803BEEA0803BE2DF2433F2DF2433FB57F743FB57F743F6C85883E6C85883E74BC2ABF74BC2ABF44C27CBF44C27CBFE0CACCBEE0CACCBEE81B0E3FE81B0E3FBBF57F3FBBF57F3F807B063F807B063F6D46DDBE6D46DDBEB3097EBFB3097EBF80E023BF80E023BF44E7993E44E7993E0308773F0308773FD1FD3D3FD1FD3D3F0EE726BE0EE726BE95146BBF95146BBFA64D54BFA64D54BF2C43B53C2C43B53CA26C5A3FA26C5A3FAC5D663FAC5D663FB714F43DB714F43D836545BF836545BF56D173BF56D173BFE62584BEE62584BEF76A2C3FF76A2C3FB8637C3FB8637C3F83A1C83E83A1C83EFAFC0FBFFAFC0FBFE5E87FBFE5E87FBF908C04BF908C04BF6D5AE13E6D5AE13ED54E7E3FD54E7E3F3521223F3521223F3A389EBE3A389EBEBC9D77BFBC9D77BF20773CBF20773CBFACD62F3EACD62F3EE4F76B3FE4F76B3F6107533F6107533F5ABFFDBC5ABFFDBCFC985BBFFC985BBF5A5E65BF5A5E65BFB512E2BDB512E2BDE4D4463FE4D4463F141E733F141E733F72877F3E72877F3E05162EBF05162EBF1C007CBF1C007CBF1F74C4BE1F74C4BE28DB113F28DB113FEDD67F3FEDD67F3FF69A023FF69A023FE869E5BEE869E5BEDE8E7EBFDE8E7EBFAA5E20BFAA5E20BF0486A23E0486A23E7D2E783F7D2E783FA8EC3A3FA8EC3A3FC2C238BEC2C238BE78D66CBF78D66CBFE0BC51BFE0BC51BF381B233D381B233DEEC05C3FEEC05C3F6E5A643F6E5A643F2A0CD03D2A0CD03D494048BF494048BFF16572BFF16572BFF8BD76BEF8BD76BE95BD2F3F95BD2F3F71977B3F71977B3FCB42C03ECB42C03E6AB613BF6AB613BFD4BF7FBFD4BF7FBFBFA600BFBFA600BFC874E93EC874E93ECBC97E3FCBC97E3FE7981E3FE7981E3F8BD0A6BE8BD0A6BE44BA78BF44BA78BF6F5E39BF6F5E39BF25AB413E25AB413E4CB06D3F4CB06D3F2A6E503F2A6E503F7E5347BD7E5347BD73E45DBF73E45DBFEE5163BFEE5163BF7301BEBD7301BEBDA9A7493FA9A7493FF2A8713FF2A8713F8BEF6D3E8BEF6D3E9F6131BF9F6131BFBB297BBFBB297BBF9C0DBCBE9C0DBCBEB68E153FB68E153F99A37F3F99A37F3FE45FFD3EE45FFD3EFA7AEDBEFA7AEDBE9DFF7EBF9DFF7EBFF6CF1CBFF6CF1CBFBA17AB3EBA17AB3E0E41793F0E41793F7FCC373F7FCC373FA48F4ABEA48F4ABE5B856EBF5B856EBF461B4FBF461B4FBFC5876B3DC5876B3D84035F3F84035F3FDF44623FDF44623FEDF2AB3DEDF2AB3DFD0A4BBFFD0A4BBF1AE770BF1AE770BF591C65BE591C65BE1A02333F1A02333FFCB67A3FFCB67A3FA7D4B73EA7D4B73E016417BF016417BF3E827FBF3E827FBF366DF9BE366DF9BE697CF13E697CF13E51307F3F51307F3FE0031B3FE0031B3F7A5BAFBE7A5BAFBED8C279BFD8C279BFDF3636BFDF3636BF1470533E1470533EA2556F3FA2556F3F3BC44D3F3BC44D3FA9DB87BDA9DB87BD1C1E60BF1C1E60BF463361BF463361BFF3E099BDF3E099BD3F6A4C3F3F6A4C3F6D20703F6D20703F8E445C3E8E445C3EFE9E34BFFE9E34BF353F7ABF353F7ABF0298B3BE0298B3BE4336193F4336193FC35B7F3FC35B7F3F8875F53E8875F53EFF78F5BEFF78F5BEE75B7FBFE75B7FBFAE3419BFAE3419BFB69BB33EB69BB33E9F3F7A3F9F3F7A3F979D343F979D343F464C5CBE464C5CBE1C2170BF1C2170BF0F694CBF0F694CBFB6F0993DB6F0993D3634613F3634613F281D603F281D603FE4CB873DE4CB873D67C54DBF67C54DBFEF546FBFEF546FBF586853BE586853BE4238363F4238363F69C2793F69C2793FC357AF3EC357AF3E72051BBF72051BBF29307FBF29307FBFED78F1BEED78F1BEAA70F93EAA70F93E5D827F3F5D827F3F6962173F6962173F57D8B7BE57D8B7BE62B77ABF62B77ABFB00033BFB00033BF0D24653E0D24653EC5E7703FC5E7703FC9094B3FC9094B3FAC02ACBDAC02ACBDCB4562BFCB4562BF8C025FBF8C025FBF36686BBD36686BBD6F1C4F3F6F1C4F3FA4846E3FA4846E3FE5874A3EE5874A3EDFCD37BFDFCD37BF9A4079BF9A4079BF0014ABBE0014ABBE86D11C3F86D11C3F70FF7E3F70FF7E3F7A77ED3E7A77ED3E5363FDBE5363FDBEB4A37FBFB4A37FBF1B8D15BF1B8D15BF4911BC3E4911BC3E1D2A7B3F1D2A7B3F3260313F3260313F3BF76DBE3BF76DBE99A971BF99A971BF71A649BF71A649BF3011BE3D3011BE3DD752633FD752633F76E35D3F76E35D3FEC33473DEC33473D506F50BF506F50BF90AF6DBF90AF6DBF62A341BE62A341BECC5F393FCC5F393FCCB9783FCCB9783FCFCCA63ECFCCA63E749A1EBF749A1EBF9AC97EBF9AC97EBF4471E9BE4471E9BE74A8003F74A8003FEABF7F3FEABF7F3FCDB4133FCDB4133F7546C0BE7546C0BECF977BBFCF977BBF25BC2FBF25BC2FBFA3C5763EA3C5763E9466723F9466723F0E3F483F0E3F483FE31BD0BDE31BD0BD535B64BF535B64BFEEBF5CBFEEBF5CBF"> : tensor<256x2xf32>
    %1 = stablehlo.constant dense<"0x0000000000000000A46A573FA46A573FB7C7683FB7C7683FC381103EC381103ECFBD41BFCFBD41BF107C75BF107C75BF8C0F8FBE8C0F8FBE4630283F4630283F95467D3F95467D3F3201D33E3201D33EF8440BBFF8440BBF5CFF7FBF5CFF7FBFD85C09BFD85C09BF2220D73E2220D73E72987D3F72987D3F4479263F4479263F106893BE106893BE251E76BF251E76BFB34040BFB34040BF6979193E6979193EC8B6693FC8B6693F102F563F102F563F150511BC150511BCE7A158BFE7A158BFFCD367BFFCD367BF388707BE388707BE0837433F0837433F0ED5743F0ED5743F2AB48A3E2AB48A3EE9E329BFE9E329BFA4EF7CBFA4EF7CBF06DECEBE06DECEBE4D2A0D3F4D2A0D3F39FA7F3F39FA7F3FF771073FF771073FC23ADBBEC23ADBBE38E57DBF38E57DBFEABE24BFEABE24BF9FBD973E9FBD973E4BBB763F4BBB763FBCBF3E3FBCBF3E3FFB6D22BEFB6D22BE28A16ABF28A16ABF31EF54BF31EF54BFA103913CA103913CD0D4593FD0D4593F9ADB663F9ADB663FE813FD3DE813FD3D56AC44BF56AC44BF222974BF222974BFFF5586BEFF5586BE24942B3F24942B3FA0937C3FA0937C3FB4B6CA3EB4B6CA3ECD0C0FBFCD0C0FBFF4EF7FBFF4EF7FBF5E8405BF5E8405BFFD50DF3EFD50DF3EE72C7E3FE72C7E3F4301233F4301233F23109CBE23109CBE7F5377BF7F5377BFF23A3DBFF23A3DBF4B5F2B3E4B5F2B3ED4866B3FD4866B3F0BAB533F0BAB533FCE81D9BCCE81D9BC5B035BBF5B035BBF97DE65BF97DE65BF4D14EBBD4D14EBBDB31D463FB31D463F5178733F5178733F23F5813E23F5813EED402DBFED402DBF8C327CBF8C327CBF518BC6BE518BC6BE6EEC103F6EEC103F8EE07F3F8EE07F3F1894033F1894033FBC62E3BEBC62E3BE7D6F7EBF7D6F7EBF574021BF574021BF865FA03E865FA03EBCE6773FBCE6773F5CB23B3F5CB23B3F2B4D34BE2B4D34BEC6676CBFC6676CBFA76252BFA76252BFD0FD103DD0FD103D822D5C3F822D5C3FF7DC643FF7DC643FFB0FD93DFB0FD93D178B47BF178B47BF9EC272BF9EC272BF56237BBE56237BBE3DEA2E3F3DEA2E3F68CC7B3F68CC7B3FF25BC23EF25BC23E28C912BF28C912BF05CC7FBF05CC7FBF2EA101BF2EA101BFED6FE73EED6FE73EF8AC7E3FF8AC7E3F2F7C1F3F2F7C1F3FB1ABA4BEB1ABA4BE007578BF007578BF03263ABF03263ABF6D373D3E6D373D3EFA436D3FFA436D3F0B16513F0B16513FCF3735BDCF3735BD3E535DBF3E535DBFC0D663BFC0D663BF4F07C7BD4F07C7BD7AF4483F7AF4483F0D08723F0D08723F5D57723E5D57723E0B9030BF0B9030BF38617BBF38617BBFAE28BEBEAE28BEBEEFA2143FEFA2143F5BB27F3F5BB27F3F5557FF3E5557FF3E7878EBBE7878EBBE58E57EBF58E57EBFD4B41DBFD4B41DBF8FF4A83E8FF4A83E49FE783F49FE783FED95383FED95383FE41D46BEE41D46BE6C1B6EBF6C1B6EBF3DC54FBF3DC54FBF2D6E593D2D6E593D8A745E3F8A745E3FF8CB623FF8CB623FA4FAB43DA4FAB43DD5594ABFD5594ABFA14871BFA14871BF888669BE888669BE4F32323F4F32323FFCF07A3FFCF07A3F99F1B93E99F1B93EBC7916BFBC7916BF90937FBF90937FBF2F67FBBE2F67FBBE4B7CEF3E4B7CEF3E9B187F3F9B187F3F4FEA1B3F4FEA1B3F093AADBE093AADBE938279BF938279BF240237BF240237BF61004F3E61004F3E18EE6E3F18EE6E3F45704E3F45704E3F2EA07DBD2EA07DBD60915FBF60915FBFA3BC61BFA3BC61BF59EAA2BD59EAA2BD21BB4B3F21BB4B3F5D84703F5D84703F04B1603E04B1603EFFD033BFFFD033BFB97B7ABFB97B7ABFC9B6B5BEC9B6B5BE844D183F844D183FA46F7F3FA46F7F3FFE71F73EFE71F73E507BF3BE507BF3BEC0467FBFC0467FBFAA1C1ABFAA1C1ABF0A7CB13E0A7CB13EDC017A3FDC017A3FAF6A353FAF6A353FB7DE57BEB7DE57BEF9BB6FBFF9BB6FBF28174DBF28174DBF8CE6903D8CE6903DB9A9603FB9A9603FC7A8603FC7A8603FC9D6903DC9D6903D57184DBF57184DBF47BB6FBF47BB6FBFFED657BEFED657BE146C353F146C353F6F017A3F6F017A3F5578B13E5578B13E3E1E1ABF3E1E1ABF9A467FBF9A467FBFD777F3BED777F3BE7375F73E7375F73EC66F7F3FC66F7F3FED4B183FED4B183F7BBAB5BE7BBAB5BE217C7ABF217C7ABF97CF33BF97CF33BFB9B8603EB9B8603E0B85703F0B85703FEFB94B3FEFB94B3F1AFAA2BD1AFAA2BD92BD61BF92BD61BF69905FBF69905FBFA2807DBDA2807DBD70714E3F70714E3F62ED6E3F62ED6E3FA4F84E3EA4F84E3E860337BF860337BF228279BF228279BF5136ADBE5136ADBEE0EB1B3FE0EB1B3F70187F3F70187F3FCD78EF3ECD78EF3EA06AFBBEA06AFBBEAD937FBFAD937FBF237816BF237816BF47F5B93E47F5B93E61F17A3F61F17A3FE430323FE430323F3A8E69BE3A8E69BE4A4971BF4A4971BF9F584ABF9F584ABF620AB53D620AB53DE3CC623FE3CC623F90735E3F90735E3F9D4E593D9D4E593D65C64FBF65C64FBFB31A6EBFB31A6EBF231646BE231646BE4C97383F4C97383FD3FD783FD3FD783FD4F0A83ED4F0A83E62B61DBF62B61DBF29E57EBF29E57EBFF674EBBEF674EBBEC15AFF3EC15AFF3E73B27F3F73B27F3F53A1143F53A1143F592CBEBE592CBEBE97617BBF97617BBF9D8E30BF9D8E30BF0B5F723E0B5F723EB208723FB208723F40F3483F40F3483F0917C7BD0917C7BDA7D763BFA7D763BF40525DBF40525DBF3C1835BD3C1835BD2F17513F2F17513F3C436D3F3C436D3FA92F3D3EA92F3D3E5E273ABF5E273ABF867478BF867478BFF3A7A4BEF3A7A4BEBA7D1F3FBA7D1F3FC5AC7E3FC5AC7E3F666CE73E666CE73EE2A201BFE2A201BF"> : tensor<256x2xf32>
    %2 = stablehlo.constant dense<0x00000000> : tensor<f32>
    %3 = stablehlo.constant dense<0x3F800000> : tensor<f32>
    %4 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.constant dense<0xF149F2CA> : tensor<f32>
    %6 = stablehlo.constant dense<0x3727C5AC> : tensor<f32>
    %7 = stablehlo.constant dense<0x42000000> : tensor<f32>
    %8 = stablehlo.constant dense<0x3EB504F3> : tensor<f32>
    %9 = stablehlo.constant dense<0> : tensor<i32>
    %10 = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.constant dense<1> : tensor<i32>
    %12 = stablehlo.dynamic_slice %arg0, %arg32, %9, sizes = [1, 32] : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x32xf32>
    %13 = stablehlo.reshape %12 : (tensor<1x32xf32>) -> tensor<32xf32>
    %14 = stablehlo.dynamic_slice %0, %arg33, %9, sizes = [1, 2] : (tensor<256x2xf32>, tensor<i32>, tensor<i32>) -> tensor<1x2xf32>
    %15 = stablehlo.reshape %14 : (tensor<1x2xf32>) -> tensor<2xf32>
    %16 = stablehlo.dynamic_slice %1, %arg33, %9, sizes = [1, 2] : (tensor<256x2xf32>, tensor<i32>, tensor<i32>) -> tensor<1x2xf32>
    %17 = stablehlo.reshape %16 : (tensor<1x2xf32>) -> tensor<2xf32>
    %18 = stablehlo.iota dim = 0 : tensor<256xi32>
    %19 = stablehlo.broadcast_in_dim %arg34, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %20 = stablehlo.compare LE, %18, %19, SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %21 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %22 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %23 = stablehlo.select %20, %21, %22 : tensor<256xi1>, tensor<256xf32>
    %24 = stablehlo.reduce(%13 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %25 = stablehlo.divide %24, %7 : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %27 = stablehlo.subtract %13, %26 : tensor<32xf32>
    %28 = stablehlo.multiply %27, %27 : tensor<32xf32>
    %29 = stablehlo.reduce(%28 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %30 = stablehlo.divide %29, %7 : tensor<f32>
    %31 = stablehlo.add %30, %6 : tensor<f32>
    %32 = stablehlo.rsqrt %31 : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %34 = stablehlo.multiply %27, %33 : tensor<32xf32>
    %35 = stablehlo.multiply %34, %arg6 : tensor<32xf32>
    %36 = stablehlo.add %35, %arg16 : tensor<32xf32>
    %37 = stablehlo.dot_general %36, %arg11, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<32x32xf32>) -> tensor<32xf32>
    %38 = stablehlo.add %37, %arg14 : tensor<32xf32>
    %39 = stablehlo.reshape %38 : (tensor<32xf32>) -> tensor<4x8xf32>
    %40 = stablehlo.dot_general %36, %arg9, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<16x32xf32>) -> tensor<16xf32>
    %41 = stablehlo.add %40, %arg13 : tensor<16xf32>
    %42 = stablehlo.reshape %41 : (tensor<16xf32>) -> tensor<2x8xf32>
    %43 = stablehlo.dot_general %36, %arg12, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<16x32xf32>) -> tensor<16xf32>
    %44 = stablehlo.add %43, %arg15 : tensor<16xf32>
    %45 = stablehlo.reshape %44 : (tensor<16xf32>) -> tensor<2x8xf32>
    %46 = stablehlo.slice %39 [0:4, 0:2] : (tensor<4x8xf32>) -> tensor<4x2xf32>
    %47 = stablehlo.slice %39 [0:4, 2:8] : (tensor<4x8xf32>) -> tensor<4x6xf32>
    %48 = stablehlo.broadcast_in_dim %15, dims = [1] : (tensor<2xf32>) -> tensor<4x2xf32>
    %49 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<2xf32>) -> tensor<4x2xf32>
    %50 = stablehlo.multiply %46, %48 : tensor<4x2xf32>
    %51 = stablehlo.slice %46 [0:4, 0:1] : (tensor<4x2xf32>) -> tensor<4x1xf32>
    %52 = stablehlo.slice %46 [0:4, 1:2] : (tensor<4x2xf32>) -> tensor<4x1xf32>
    %53 = stablehlo.negate %52 : tensor<4x1xf32>
    %54 = stablehlo.concatenate %53, %51, dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
    %55 = stablehlo.multiply %54, %49 : tensor<4x2xf32>
    %56 = stablehlo.add %50, %55 : tensor<4x2xf32>
    %57 = stablehlo.concatenate %56, %47, dim = 1 : (tensor<4x2xf32>, tensor<4x6xf32>) -> tensor<4x8xf32>
    %58 = stablehlo.slice %42 [0:2, 0:2] : (tensor<2x8xf32>) -> tensor<2x2xf32>
    %59 = stablehlo.slice %42 [0:2, 2:8] : (tensor<2x8xf32>) -> tensor<2x6xf32>
    %60 = stablehlo.broadcast_in_dim %15, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %61 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %62 = stablehlo.multiply %58, %60 : tensor<2x2xf32>
    %63 = stablehlo.slice %58 [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %64 = stablehlo.slice %58 [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %65 = stablehlo.negate %64 : tensor<2x1xf32>
    %66 = stablehlo.concatenate %65, %63, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
    %67 = stablehlo.multiply %66, %61 : tensor<2x2xf32>
    %68 = stablehlo.add %62, %67 : tensor<2x2xf32>
    %69 = stablehlo.concatenate %68, %59, dim = 1 : (tensor<2x2xf32>, tensor<2x6xf32>) -> tensor<2x8xf32>
    %70 = stablehlo.reshape %69 : (tensor<2x8xf32>) -> tensor<1x1x2x8xf32>
    %71 = stablehlo.dynamic_update_slice %arg35, %70, %10, %arg34, %9, %9 : (tensor<2x256x2x8xf32>, tensor<1x1x2x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x256x2x8xf32>
    %72 = stablehlo.reshape %45 : (tensor<2x8xf32>) -> tensor<1x1x2x8xf32>
    %73 = stablehlo.dynamic_update_slice %arg36, %72, %10, %arg34, %9, %9 : (tensor<2x256x2x8xf32>, tensor<1x1x2x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x256x2x8xf32>
    %74 = stablehlo.slice %71 [0:1, 0:256, 0:2, 0:8] : (tensor<2x256x2x8xf32>) -> tensor<1x256x2x8xf32>
    %75 = stablehlo.reshape %74 : (tensor<1x256x2x8xf32>) -> tensor<256x2x8xf32>
    %76 = stablehlo.slice %73 [0:1, 0:256, 0:2, 0:8] : (tensor<2x256x2x8xf32>) -> tensor<1x256x2x8xf32>
    %77 = stablehlo.reshape %76 : (tensor<1x256x2x8xf32>) -> tensor<256x2x8xf32>
    %78 = stablehlo.reshape %57 : (tensor<4x8xf32>) -> tensor<2x2x8xf32>
    %79 = stablehlo.dot_general %78, %75, batching_dims = [0] x [1], contracting_dims = [2] x [2] : (tensor<2x2x8xf32>, tensor<256x2x8xf32>) -> tensor<2x2x256xf32>
    %80 = stablehlo.reshape %79 : (tensor<2x2x256xf32>) -> tensor<4x256xf32>
    %81 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4x256xf32>
    %82 = stablehlo.multiply %80, %81 : tensor<4x256xf32>
    %83 = stablehlo.broadcast_in_dim %23, dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
    %84 = stablehlo.add %82, %83 : tensor<4x256xf32>
    %85 = stablehlo.reduce(%84 init: %4) applies stablehlo.maximum across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %86 = stablehlo.broadcast_in_dim %85, dims = [0] : (tensor<4xf32>) -> tensor<4x256xf32>
    %87 = stablehlo.subtract %84, %86 : tensor<4x256xf32>
    %88 = stablehlo.exponential %87 : tensor<4x256xf32>
    %89 = stablehlo.reduce(%88 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [0] : (tensor<4xf32>) -> tensor<4x256xf32>
    %91 = stablehlo.divide %88, %90 : tensor<4x256xf32>
    %92 = stablehlo.reshape %91 : (tensor<4x256xf32>) -> tensor<2x2x256xf32>
    %93 = stablehlo.dot_general %92, %77, batching_dims = [0] x [1], contracting_dims = [2] x [0] : (tensor<2x2x256xf32>, tensor<256x2x8xf32>) -> tensor<2x2x8xf32>
    %94 = stablehlo.reshape %93 : (tensor<2x2x8xf32>) -> tensor<4x8xf32>
    %95 = stablehlo.reshape %94 : (tensor<4x8xf32>) -> tensor<32xf32>
    %96 = stablehlo.dot_general %95, %arg10, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<32x32xf32>) -> tensor<32xf32>
    %97 = stablehlo.add %13, %96 : tensor<32xf32>
    %98 = stablehlo.reduce(%97 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %99 = stablehlo.divide %98, %7 : tensor<f32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %101 = stablehlo.subtract %97, %100 : tensor<32xf32>
    %102 = stablehlo.multiply %101, %101 : tensor<32xf32>
    %103 = stablehlo.reduce(%102 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %104 = stablehlo.divide %103, %7 : tensor<f32>
    %105 = stablehlo.add %104, %6 : tensor<f32>
    %106 = stablehlo.rsqrt %105 : tensor<f32>
    %107 = stablehlo.broadcast_in_dim %106, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %108 = stablehlo.multiply %101, %107 : tensor<32xf32>
    %109 = stablehlo.multiply %108, %arg7 : tensor<32xf32>
    %110 = stablehlo.add %109, %arg17 : tensor<32xf32>
    %111 = stablehlo.dot_general %110, %arg5, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<64x32xf32>) -> tensor<64xf32>
    %112 = stablehlo.dot_general %110, %arg8, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<64x32xf32>) -> tensor<64xf32>
    %113 = stablehlo.negate %111 : tensor<64xf32>
    %114 = stablehlo.exponential %113 : tensor<64xf32>
    %115 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %116 = stablehlo.add %115, %114 : tensor<64xf32>
    %117 = stablehlo.divide %115, %116 : tensor<64xf32>
    %118 = stablehlo.multiply %111, %117 : tensor<64xf32>
    %119 = stablehlo.multiply %118, %112 : tensor<64xf32>
    %120 = stablehlo.dot_general %119, %arg4, contracting_dims = [0] x [1] : (tensor<64xf32>, tensor<32x64xf32>) -> tensor<32xf32>
    %121 = stablehlo.add %97, %120 : tensor<32xf32>
    %122 = stablehlo.reduce(%121 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %123 = stablehlo.divide %122, %7 : tensor<f32>
    %124 = stablehlo.broadcast_in_dim %123, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %125 = stablehlo.subtract %121, %124 : tensor<32xf32>
    %126 = stablehlo.multiply %125, %125 : tensor<32xf32>
    %127 = stablehlo.reduce(%126 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %128 = stablehlo.divide %127, %7 : tensor<f32>
    %129 = stablehlo.add %128, %6 : tensor<f32>
    %130 = stablehlo.rsqrt %129 : tensor<f32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %132 = stablehlo.multiply %125, %131 : tensor<32xf32>
    %133 = stablehlo.multiply %132, %arg20 : tensor<32xf32>
    %134 = stablehlo.add %133, %arg30 : tensor<32xf32>
    %135 = stablehlo.dot_general %134, %arg25, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<32x32xf32>) -> tensor<32xf32>
    %136 = stablehlo.add %135, %arg28 : tensor<32xf32>
    %137 = stablehlo.reshape %136 : (tensor<32xf32>) -> tensor<4x8xf32>
    %138 = stablehlo.dot_general %134, %arg23, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<16x32xf32>) -> tensor<16xf32>
    %139 = stablehlo.add %138, %arg27 : tensor<16xf32>
    %140 = stablehlo.reshape %139 : (tensor<16xf32>) -> tensor<2x8xf32>
    %141 = stablehlo.dot_general %134, %arg26, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<16x32xf32>) -> tensor<16xf32>
    %142 = stablehlo.add %141, %arg29 : tensor<16xf32>
    %143 = stablehlo.reshape %142 : (tensor<16xf32>) -> tensor<2x8xf32>
    %144 = stablehlo.slice %137 [0:4, 0:2] : (tensor<4x8xf32>) -> tensor<4x2xf32>
    %145 = stablehlo.slice %137 [0:4, 2:8] : (tensor<4x8xf32>) -> tensor<4x6xf32>
    %146 = stablehlo.broadcast_in_dim %15, dims = [1] : (tensor<2xf32>) -> tensor<4x2xf32>
    %147 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<2xf32>) -> tensor<4x2xf32>
    %148 = stablehlo.multiply %144, %146 : tensor<4x2xf32>
    %149 = stablehlo.slice %144 [0:4, 0:1] : (tensor<4x2xf32>) -> tensor<4x1xf32>
    %150 = stablehlo.slice %144 [0:4, 1:2] : (tensor<4x2xf32>) -> tensor<4x1xf32>
    %151 = stablehlo.negate %150 : tensor<4x1xf32>
    %152 = stablehlo.concatenate %151, %149, dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
    %153 = stablehlo.multiply %152, %147 : tensor<4x2xf32>
    %154 = stablehlo.add %148, %153 : tensor<4x2xf32>
    %155 = stablehlo.concatenate %154, %145, dim = 1 : (tensor<4x2xf32>, tensor<4x6xf32>) -> tensor<4x8xf32>
    %156 = stablehlo.slice %140 [0:2, 0:2] : (tensor<2x8xf32>) -> tensor<2x2xf32>
    %157 = stablehlo.slice %140 [0:2, 2:8] : (tensor<2x8xf32>) -> tensor<2x6xf32>
    %158 = stablehlo.broadcast_in_dim %15, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %159 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %160 = stablehlo.multiply %156, %158 : tensor<2x2xf32>
    %161 = stablehlo.slice %156 [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %162 = stablehlo.slice %156 [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %163 = stablehlo.negate %162 : tensor<2x1xf32>
    %164 = stablehlo.concatenate %163, %161, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
    %165 = stablehlo.multiply %164, %159 : tensor<2x2xf32>
    %166 = stablehlo.add %160, %165 : tensor<2x2xf32>
    %167 = stablehlo.concatenate %166, %157, dim = 1 : (tensor<2x2xf32>, tensor<2x6xf32>) -> tensor<2x8xf32>
    %168 = stablehlo.reshape %167 : (tensor<2x8xf32>) -> tensor<1x1x2x8xf32>
    %169 = stablehlo.dynamic_update_slice %71, %168, %11, %arg34, %9, %9 : (tensor<2x256x2x8xf32>, tensor<1x1x2x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x256x2x8xf32>
    %170 = stablehlo.reshape %143 : (tensor<2x8xf32>) -> tensor<1x1x2x8xf32>
    %171 = stablehlo.dynamic_update_slice %73, %170, %11, %arg34, %9, %9 : (tensor<2x256x2x8xf32>, tensor<1x1x2x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x256x2x8xf32>
    %172 = stablehlo.slice %169 [1:2, 0:256, 0:2, 0:8] : (tensor<2x256x2x8xf32>) -> tensor<1x256x2x8xf32>
    %173 = stablehlo.reshape %172 : (tensor<1x256x2x8xf32>) -> tensor<256x2x8xf32>
    %174 = stablehlo.slice %171 [1:2, 0:256, 0:2, 0:8] : (tensor<2x256x2x8xf32>) -> tensor<1x256x2x8xf32>
    %175 = stablehlo.reshape %174 : (tensor<1x256x2x8xf32>) -> tensor<256x2x8xf32>
    %176 = stablehlo.reshape %155 : (tensor<4x8xf32>) -> tensor<2x2x8xf32>
    %177 = stablehlo.dot_general %176, %173, batching_dims = [0] x [1], contracting_dims = [2] x [2] : (tensor<2x2x8xf32>, tensor<256x2x8xf32>) -> tensor<2x2x256xf32>
    %178 = stablehlo.reshape %177 : (tensor<2x2x256xf32>) -> tensor<4x256xf32>
    %179 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4x256xf32>
    %180 = stablehlo.multiply %178, %179 : tensor<4x256xf32>
    %181 = stablehlo.broadcast_in_dim %23, dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
    %182 = stablehlo.add %180, %181 : tensor<4x256xf32>
    %183 = stablehlo.reduce(%182 init: %4) applies stablehlo.maximum across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %184 = stablehlo.broadcast_in_dim %183, dims = [0] : (tensor<4xf32>) -> tensor<4x256xf32>
    %185 = stablehlo.subtract %182, %184 : tensor<4x256xf32>
    %186 = stablehlo.exponential %185 : tensor<4x256xf32>
    %187 = stablehlo.reduce(%186 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0] : (tensor<4xf32>) -> tensor<4x256xf32>
    %189 = stablehlo.divide %186, %188 : tensor<4x256xf32>
    %190 = stablehlo.reshape %189 : (tensor<4x256xf32>) -> tensor<2x2x256xf32>
    %191 = stablehlo.dot_general %190, %175, batching_dims = [0] x [1], contracting_dims = [2] x [0] : (tensor<2x2x256xf32>, tensor<256x2x8xf32>) -> tensor<2x2x8xf32>
    %192 = stablehlo.reshape %191 : (tensor<2x2x8xf32>) -> tensor<4x8xf32>
    %193 = stablehlo.reshape %192 : (tensor<4x8xf32>) -> tensor<32xf32>
    %194 = stablehlo.dot_general %193, %arg24, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<32x32xf32>) -> tensor<32xf32>
    %195 = stablehlo.add %121, %194 : tensor<32xf32>
    %196 = stablehlo.reduce(%195 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %197 = stablehlo.divide %196, %7 : tensor<f32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %199 = stablehlo.subtract %195, %198 : tensor<32xf32>
    %200 = stablehlo.multiply %199, %199 : tensor<32xf32>
    %201 = stablehlo.reduce(%200 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %202 = stablehlo.divide %201, %7 : tensor<f32>
    %203 = stablehlo.add %202, %6 : tensor<f32>
    %204 = stablehlo.rsqrt %203 : tensor<f32>
    %205 = stablehlo.broadcast_in_dim %204, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %206 = stablehlo.multiply %199, %205 : tensor<32xf32>
    %207 = stablehlo.multiply %206, %arg21 : tensor<32xf32>
    %208 = stablehlo.add %207, %arg31 : tensor<32xf32>
    %209 = stablehlo.dot_general %208, %arg19, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<64x32xf32>) -> tensor<64xf32>
    %210 = stablehlo.dot_general %208, %arg22, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<64x32xf32>) -> tensor<64xf32>
    %211 = stablehlo.negate %209 : tensor<64xf32>
    %212 = stablehlo.exponential %211 : tensor<64xf32>
    %213 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %214 = stablehlo.add %213, %212 : tensor<64xf32>
    %215 = stablehlo.divide %213, %214 : tensor<64xf32>
    %216 = stablehlo.multiply %209, %215 : tensor<64xf32>
    %217 = stablehlo.multiply %216, %210 : tensor<64xf32>
    %218 = stablehlo.dot_general %217, %arg18, contracting_dims = [0] x [1] : (tensor<64xf32>, tensor<32x64xf32>) -> tensor<32xf32>
    %219 = stablehlo.add %195, %218 : tensor<32xf32>
    %220 = stablehlo.reduce(%219 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %221 = stablehlo.divide %220, %7 : tensor<f32>
    %222 = stablehlo.broadcast_in_dim %221, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %223 = stablehlo.subtract %219, %222 : tensor<32xf32>
    %224 = stablehlo.multiply %223, %223 : tensor<32xf32>
    %225 = stablehlo.reduce(%224 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %226 = stablehlo.divide %225, %7 : tensor<f32>
    %227 = stablehlo.add %226, %6 : tensor<f32>
    %228 = stablehlo.rsqrt %227 : tensor<f32>
    %229 = stablehlo.broadcast_in_dim %228, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %230 = stablehlo.multiply %223, %229 : tensor<32xf32>
    %231 = stablehlo.multiply %230, %arg1 : tensor<32xf32>
    %232 = stablehlo.add %231, %arg2 : tensor<32xf32>
    %233 = stablehlo.dot_general %232, %arg3, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<32x32xf32>) -> tensor<32xf32>
    return %233, %169, %171 : tensor<32xf32>, tensor<2x256x2x8xf32>, tensor<2x256x2x8xf32>
  }
}
