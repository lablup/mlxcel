module @prefill {
  func.func public @main(%arg0: tensor<32x32xf32> loc("params['embed']"), %arg1: tensor<32xf32> loc("params['final_norm']"), %arg2: tensor<32xf32> loc("params['final_norm_bias']"), %arg3: tensor<32x32xf32> loc("params['lm_head']"), %arg4: tensor<32x64xf32> loc("params['layers'][0]['down']"), %arg5: tensor<64x32xf32> loc("params['layers'][0]['gate']"), %arg6: tensor<32xf32> loc("params['layers'][0]['in_ln']"), %arg7: tensor<32xf32> loc("params['layers'][0]['post_ln']"), %arg8: tensor<64x32xf32> loc("params['layers'][0]['up']"), %arg9: tensor<16x32xf32> loc("params['layers'][0]['wk']"), %arg10: tensor<32x32xf32> loc("params['layers'][0]['wo']"), %arg11: tensor<32x32xf32> loc("params['layers'][0]['wq']"), %arg12: tensor<16x32xf32> loc("params['layers'][0]['wv']"), %arg13: tensor<16xf32> loc("params['layers'][0]['bk']"), %arg14: tensor<32xf32> loc("params['layers'][0]['bq']"), %arg15: tensor<16xf32> loc("params['layers'][0]['bv']"), %arg16: tensor<32xf32> loc("params['layers'][0]['in_ln_bias']"), %arg17: tensor<32xf32> loc("params['layers'][0]['post_ln_bias']"), %arg18: tensor<32x64xf32> loc("params['layers'][1]['down']"), %arg19: tensor<64x32xf32> loc("params['layers'][1]['gate']"), %arg20: tensor<32xf32> loc("params['layers'][1]['in_ln']"), %arg21: tensor<32xf32> loc("params['layers'][1]['post_ln']"), %arg22: tensor<64x32xf32> loc("params['layers'][1]['up']"), %arg23: tensor<16x32xf32> loc("params['layers'][1]['wk']"), %arg24: tensor<32x32xf32> loc("params['layers'][1]['wo']"), %arg25: tensor<32x32xf32> loc("params['layers'][1]['wq']"), %arg26: tensor<16x32xf32> loc("params['layers'][1]['wv']"), %arg27: tensor<16xf32> loc("params['layers'][1]['bk']"), %arg28: tensor<32xf32> loc("params['layers'][1]['bq']"), %arg29: tensor<16xf32> loc("params['layers'][1]['bv']"), %arg30: tensor<32xf32> loc("params['layers'][1]['in_ln_bias']"), %arg31: tensor<32xf32> loc("params['layers'][1]['post_ln_bias']"), %arg32: tensor<256xi32> loc("tokens"), %arg33: tensor<256xi32> loc("positions"), %arg34: tensor<i32> loc("real_len")) -> (tensor<32xf32>, tensor<2x256x2x8xf32>, tensor<2x256x2x8xf32>) {
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
    %12 = stablehlo.reshape %arg32 : (tensor<256xi32>) -> tensor<256x1xi32>
    %13 = "stablehlo.gather"(%arg0, %12) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 32>}> : (tensor<32x32xf32>, tensor<256x1xi32>) -> tensor<256x32xf32>
    %14 = stablehlo.reshape %arg33 : (tensor<256xi32>) -> tensor<256x1xi32>
    %15 = "stablehlo.gather"(%0, %14) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 2>}> : (tensor<256x2xf32>, tensor<256x1xi32>) -> tensor<256x2xf32>
    %16 = "stablehlo.gather"(%1, %14) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 2>}> : (tensor<256x2xf32>, tensor<256x1xi32>) -> tensor<256x2xf32>
    %17 = stablehlo.iota dim = 0 : tensor<256xi32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<256xi32>) -> tensor<256x256xi32>
    %19 = stablehlo.iota dim = 0 : tensor<256xi32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [1] : (tensor<256xi32>) -> tensor<256x256xi32>
    %21 = stablehlo.compare LE, %20, %18, SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %22 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %23 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %24 = stablehlo.select %21, %22, %23 : tensor<256x256xi1>, tensor<256x256xf32>
    %25 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x256x2x8xf32>
    %26 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x256x2x8xf32>
    %27 = stablehlo.reduce(%13 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<256x32xf32>, tensor<f32>) -> tensor<256xf32>
    %28 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %29 = stablehlo.divide %27, %28 : tensor<256xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0] : (tensor<256xf32>) -> tensor<256x32xf32>
    %31 = stablehlo.subtract %13, %30 : tensor<256x32xf32>
    %32 = stablehlo.multiply %31, %31 : tensor<256x32xf32>
    %33 = stablehlo.reduce(%32 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<256x32xf32>, tensor<f32>) -> tensor<256xf32>
    %34 = stablehlo.divide %33, %28 : tensor<256xf32>
    %35 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %36 = stablehlo.add %34, %35 : tensor<256xf32>
    %37 = stablehlo.rsqrt %36 : tensor<256xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0] : (tensor<256xf32>) -> tensor<256x32xf32>
    %39 = stablehlo.multiply %31, %38 : tensor<256x32xf32>
    %40 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %41 = stablehlo.multiply %39, %40 : tensor<256x32xf32>
    %42 = stablehlo.broadcast_in_dim %arg16, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %43 = stablehlo.add %41, %42 : tensor<256x32xf32>
    %44 = stablehlo.dot_general %43, %arg11, contracting_dims = [1] x [1] : (tensor<256x32xf32>, tensor<32x32xf32>) -> tensor<256x32xf32>
    %45 = stablehlo.broadcast_in_dim %arg14, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %46 = stablehlo.add %44, %45 : tensor<256x32xf32>
    %47 = stablehlo.reshape %46 : (tensor<256x32xf32>) -> tensor<256x4x8xf32>
    %48 = stablehlo.dot_general %43, %arg9, contracting_dims = [1] x [1] : (tensor<256x32xf32>, tensor<16x32xf32>) -> tensor<256x16xf32>
    %49 = stablehlo.broadcast_in_dim %arg13, dims = [1] : (tensor<16xf32>) -> tensor<256x16xf32>
    %50 = stablehlo.add %48, %49 : tensor<256x16xf32>
    %51 = stablehlo.reshape %50 : (tensor<256x16xf32>) -> tensor<256x2x8xf32>
    %52 = stablehlo.dot_general %43, %arg12, contracting_dims = [1] x [1] : (tensor<256x32xf32>, tensor<16x32xf32>) -> tensor<256x16xf32>
    %53 = stablehlo.broadcast_in_dim %arg15, dims = [1] : (tensor<16xf32>) -> tensor<256x16xf32>
    %54 = stablehlo.add %52, %53 : tensor<256x16xf32>
    %55 = stablehlo.reshape %54 : (tensor<256x16xf32>) -> tensor<256x2x8xf32>
    %56 = stablehlo.slice %47 [0:256, 0:4, 0:2] : (tensor<256x4x8xf32>) -> tensor<256x4x2xf32>
    %57 = stablehlo.slice %47 [0:256, 0:4, 2:8] : (tensor<256x4x8xf32>) -> tensor<256x4x6xf32>
    %58 = stablehlo.broadcast_in_dim %15, dims = [0, 2] : (tensor<256x2xf32>) -> tensor<256x4x2xf32>
    %59 = stablehlo.broadcast_in_dim %16, dims = [0, 2] : (tensor<256x2xf32>) -> tensor<256x4x2xf32>
    %60 = stablehlo.multiply %56, %58 : tensor<256x4x2xf32>
    %61 = stablehlo.slice %56 [0:256, 0:4, 0:1] : (tensor<256x4x2xf32>) -> tensor<256x4x1xf32>
    %62 = stablehlo.slice %56 [0:256, 0:4, 1:2] : (tensor<256x4x2xf32>) -> tensor<256x4x1xf32>
    %63 = stablehlo.negate %62 : tensor<256x4x1xf32>
    %64 = stablehlo.concatenate %63, %61, dim = 2 : (tensor<256x4x1xf32>, tensor<256x4x1xf32>) -> tensor<256x4x2xf32>
    %65 = stablehlo.multiply %64, %59 : tensor<256x4x2xf32>
    %66 = stablehlo.add %60, %65 : tensor<256x4x2xf32>
    %67 = stablehlo.concatenate %66, %57, dim = 2 : (tensor<256x4x2xf32>, tensor<256x4x6xf32>) -> tensor<256x4x8xf32>
    %68 = stablehlo.slice %51 [0:256, 0:2, 0:2] : (tensor<256x2x8xf32>) -> tensor<256x2x2xf32>
    %69 = stablehlo.slice %51 [0:256, 0:2, 2:8] : (tensor<256x2x8xf32>) -> tensor<256x2x6xf32>
    %70 = stablehlo.broadcast_in_dim %15, dims = [0, 2] : (tensor<256x2xf32>) -> tensor<256x2x2xf32>
    %71 = stablehlo.broadcast_in_dim %16, dims = [0, 2] : (tensor<256x2xf32>) -> tensor<256x2x2xf32>
    %72 = stablehlo.multiply %68, %70 : tensor<256x2x2xf32>
    %73 = stablehlo.slice %68 [0:256, 0:2, 0:1] : (tensor<256x2x2xf32>) -> tensor<256x2x1xf32>
    %74 = stablehlo.slice %68 [0:256, 0:2, 1:2] : (tensor<256x2x2xf32>) -> tensor<256x2x1xf32>
    %75 = stablehlo.negate %74 : tensor<256x2x1xf32>
    %76 = stablehlo.concatenate %75, %73, dim = 2 : (tensor<256x2x1xf32>, tensor<256x2x1xf32>) -> tensor<256x2x2xf32>
    %77 = stablehlo.multiply %76, %71 : tensor<256x2x2xf32>
    %78 = stablehlo.add %72, %77 : tensor<256x2x2xf32>
    %79 = stablehlo.concatenate %78, %69, dim = 2 : (tensor<256x2x2xf32>, tensor<256x2x6xf32>) -> tensor<256x2x8xf32>
    %80 = stablehlo.reshape %79 : (tensor<256x2x8xf32>) -> tensor<1x256x2x8xf32>
    %81 = stablehlo.dynamic_update_slice %25, %80, %10, %9, %9, %9 : (tensor<2x256x2x8xf32>, tensor<1x256x2x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x256x2x8xf32>
    %82 = stablehlo.reshape %55 : (tensor<256x2x8xf32>) -> tensor<1x256x2x8xf32>
    %83 = stablehlo.dynamic_update_slice %26, %82, %10, %9, %9, %9 : (tensor<2x256x2x8xf32>, tensor<1x256x2x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x256x2x8xf32>
    %84 = stablehlo.reshape %67 : (tensor<256x4x8xf32>) -> tensor<256x2x2x8xf32>
    %85 = stablehlo.dot_general %84, %79, batching_dims = [1] x [1], contracting_dims = [3] x [2] : (tensor<256x2x2x8xf32>, tensor<256x2x8xf32>) -> tensor<2x256x2x256xf32>
    %86 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x256x2x256xf32>
    %87 = stablehlo.multiply %85, %86 : tensor<2x256x2x256xf32>
    %88 = stablehlo.broadcast_in_dim %24, dims = [1, 3] : (tensor<256x256xf32>) -> tensor<2x256x2x256xf32>
    %89 = stablehlo.add %87, %88 : tensor<2x256x2x256xf32>
    %90 = stablehlo.reduce(%89 init: %4) applies stablehlo.maximum across dimensions = [3] : (tensor<2x256x2x256xf32>, tensor<f32>) -> tensor<2x256x2xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2] : (tensor<2x256x2xf32>) -> tensor<2x256x2x256xf32>
    %92 = stablehlo.subtract %89, %91 : tensor<2x256x2x256xf32>
    %93 = stablehlo.exponential %92 : tensor<2x256x2x256xf32>
    %94 = stablehlo.reduce(%93 init: %2) applies stablehlo.add across dimensions = [3] : (tensor<2x256x2x256xf32>, tensor<f32>) -> tensor<2x256x2xf32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2] : (tensor<2x256x2xf32>) -> tensor<2x256x2x256xf32>
    %96 = stablehlo.divide %93, %95 : tensor<2x256x2x256xf32>
    %97 = stablehlo.dot_general %96, %55, batching_dims = [0] x [1], contracting_dims = [3] x [0] : (tensor<2x256x2x256xf32>, tensor<256x2x8xf32>) -> tensor<2x256x2x8xf32>
    %98 = stablehlo.transpose %97, dims = [1, 0, 2, 3] : (tensor<2x256x2x8xf32>) -> tensor<256x2x2x8xf32>
    %99 = stablehlo.reshape %98 : (tensor<256x2x2x8xf32>) -> tensor<256x32xf32>
    %100 = stablehlo.dot_general %99, %arg10, contracting_dims = [1] x [1] : (tensor<256x32xf32>, tensor<32x32xf32>) -> tensor<256x32xf32>
    %101 = stablehlo.add %13, %100 : tensor<256x32xf32>
    %102 = stablehlo.reduce(%101 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<256x32xf32>, tensor<f32>) -> tensor<256xf32>
    %103 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %104 = stablehlo.divide %102, %103 : tensor<256xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [0] : (tensor<256xf32>) -> tensor<256x32xf32>
    %106 = stablehlo.subtract %101, %105 : tensor<256x32xf32>
    %107 = stablehlo.multiply %106, %106 : tensor<256x32xf32>
    %108 = stablehlo.reduce(%107 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<256x32xf32>, tensor<f32>) -> tensor<256xf32>
    %109 = stablehlo.divide %108, %103 : tensor<256xf32>
    %110 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %111 = stablehlo.add %109, %110 : tensor<256xf32>
    %112 = stablehlo.rsqrt %111 : tensor<256xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [0] : (tensor<256xf32>) -> tensor<256x32xf32>
    %114 = stablehlo.multiply %106, %113 : tensor<256x32xf32>
    %115 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %116 = stablehlo.multiply %114, %115 : tensor<256x32xf32>
    %117 = stablehlo.broadcast_in_dim %arg17, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %118 = stablehlo.add %116, %117 : tensor<256x32xf32>
    %119 = stablehlo.dot_general %118, %arg5, contracting_dims = [1] x [1] : (tensor<256x32xf32>, tensor<64x32xf32>) -> tensor<256x64xf32>
    %120 = stablehlo.dot_general %118, %arg8, contracting_dims = [1] x [1] : (tensor<256x32xf32>, tensor<64x32xf32>) -> tensor<256x64xf32>
    %121 = stablehlo.negate %119 : tensor<256x64xf32>
    %122 = stablehlo.exponential %121 : tensor<256x64xf32>
    %123 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<256x64xf32>
    %124 = stablehlo.add %123, %122 : tensor<256x64xf32>
    %125 = stablehlo.divide %123, %124 : tensor<256x64xf32>
    %126 = stablehlo.multiply %119, %125 : tensor<256x64xf32>
    %127 = stablehlo.multiply %126, %120 : tensor<256x64xf32>
    %128 = stablehlo.dot_general %127, %arg4, contracting_dims = [1] x [1] : (tensor<256x64xf32>, tensor<32x64xf32>) -> tensor<256x32xf32>
    %129 = stablehlo.add %101, %128 : tensor<256x32xf32>
    %130 = stablehlo.reduce(%129 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<256x32xf32>, tensor<f32>) -> tensor<256xf32>
    %131 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %132 = stablehlo.divide %130, %131 : tensor<256xf32>
    %133 = stablehlo.broadcast_in_dim %132, dims = [0] : (tensor<256xf32>) -> tensor<256x32xf32>
    %134 = stablehlo.subtract %129, %133 : tensor<256x32xf32>
    %135 = stablehlo.multiply %134, %134 : tensor<256x32xf32>
    %136 = stablehlo.reduce(%135 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<256x32xf32>, tensor<f32>) -> tensor<256xf32>
    %137 = stablehlo.divide %136, %131 : tensor<256xf32>
    %138 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %139 = stablehlo.add %137, %138 : tensor<256xf32>
    %140 = stablehlo.rsqrt %139 : tensor<256xf32>
    %141 = stablehlo.broadcast_in_dim %140, dims = [0] : (tensor<256xf32>) -> tensor<256x32xf32>
    %142 = stablehlo.multiply %134, %141 : tensor<256x32xf32>
    %143 = stablehlo.broadcast_in_dim %arg20, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %144 = stablehlo.multiply %142, %143 : tensor<256x32xf32>
    %145 = stablehlo.broadcast_in_dim %arg30, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %146 = stablehlo.add %144, %145 : tensor<256x32xf32>
    %147 = stablehlo.dot_general %146, %arg25, contracting_dims = [1] x [1] : (tensor<256x32xf32>, tensor<32x32xf32>) -> tensor<256x32xf32>
    %148 = stablehlo.broadcast_in_dim %arg28, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %149 = stablehlo.add %147, %148 : tensor<256x32xf32>
    %150 = stablehlo.reshape %149 : (tensor<256x32xf32>) -> tensor<256x4x8xf32>
    %151 = stablehlo.dot_general %146, %arg23, contracting_dims = [1] x [1] : (tensor<256x32xf32>, tensor<16x32xf32>) -> tensor<256x16xf32>
    %152 = stablehlo.broadcast_in_dim %arg27, dims = [1] : (tensor<16xf32>) -> tensor<256x16xf32>
    %153 = stablehlo.add %151, %152 : tensor<256x16xf32>
    %154 = stablehlo.reshape %153 : (tensor<256x16xf32>) -> tensor<256x2x8xf32>
    %155 = stablehlo.dot_general %146, %arg26, contracting_dims = [1] x [1] : (tensor<256x32xf32>, tensor<16x32xf32>) -> tensor<256x16xf32>
    %156 = stablehlo.broadcast_in_dim %arg29, dims = [1] : (tensor<16xf32>) -> tensor<256x16xf32>
    %157 = stablehlo.add %155, %156 : tensor<256x16xf32>
    %158 = stablehlo.reshape %157 : (tensor<256x16xf32>) -> tensor<256x2x8xf32>
    %159 = stablehlo.slice %150 [0:256, 0:4, 0:2] : (tensor<256x4x8xf32>) -> tensor<256x4x2xf32>
    %160 = stablehlo.slice %150 [0:256, 0:4, 2:8] : (tensor<256x4x8xf32>) -> tensor<256x4x6xf32>
    %161 = stablehlo.broadcast_in_dim %15, dims = [0, 2] : (tensor<256x2xf32>) -> tensor<256x4x2xf32>
    %162 = stablehlo.broadcast_in_dim %16, dims = [0, 2] : (tensor<256x2xf32>) -> tensor<256x4x2xf32>
    %163 = stablehlo.multiply %159, %161 : tensor<256x4x2xf32>
    %164 = stablehlo.slice %159 [0:256, 0:4, 0:1] : (tensor<256x4x2xf32>) -> tensor<256x4x1xf32>
    %165 = stablehlo.slice %159 [0:256, 0:4, 1:2] : (tensor<256x4x2xf32>) -> tensor<256x4x1xf32>
    %166 = stablehlo.negate %165 : tensor<256x4x1xf32>
    %167 = stablehlo.concatenate %166, %164, dim = 2 : (tensor<256x4x1xf32>, tensor<256x4x1xf32>) -> tensor<256x4x2xf32>
    %168 = stablehlo.multiply %167, %162 : tensor<256x4x2xf32>
    %169 = stablehlo.add %163, %168 : tensor<256x4x2xf32>
    %170 = stablehlo.concatenate %169, %160, dim = 2 : (tensor<256x4x2xf32>, tensor<256x4x6xf32>) -> tensor<256x4x8xf32>
    %171 = stablehlo.slice %154 [0:256, 0:2, 0:2] : (tensor<256x2x8xf32>) -> tensor<256x2x2xf32>
    %172 = stablehlo.slice %154 [0:256, 0:2, 2:8] : (tensor<256x2x8xf32>) -> tensor<256x2x6xf32>
    %173 = stablehlo.broadcast_in_dim %15, dims = [0, 2] : (tensor<256x2xf32>) -> tensor<256x2x2xf32>
    %174 = stablehlo.broadcast_in_dim %16, dims = [0, 2] : (tensor<256x2xf32>) -> tensor<256x2x2xf32>
    %175 = stablehlo.multiply %171, %173 : tensor<256x2x2xf32>
    %176 = stablehlo.slice %171 [0:256, 0:2, 0:1] : (tensor<256x2x2xf32>) -> tensor<256x2x1xf32>
    %177 = stablehlo.slice %171 [0:256, 0:2, 1:2] : (tensor<256x2x2xf32>) -> tensor<256x2x1xf32>
    %178 = stablehlo.negate %177 : tensor<256x2x1xf32>
    %179 = stablehlo.concatenate %178, %176, dim = 2 : (tensor<256x2x1xf32>, tensor<256x2x1xf32>) -> tensor<256x2x2xf32>
    %180 = stablehlo.multiply %179, %174 : tensor<256x2x2xf32>
    %181 = stablehlo.add %175, %180 : tensor<256x2x2xf32>
    %182 = stablehlo.concatenate %181, %172, dim = 2 : (tensor<256x2x2xf32>, tensor<256x2x6xf32>) -> tensor<256x2x8xf32>
    %183 = stablehlo.reshape %182 : (tensor<256x2x8xf32>) -> tensor<1x256x2x8xf32>
    %184 = stablehlo.dynamic_update_slice %81, %183, %11, %9, %9, %9 : (tensor<2x256x2x8xf32>, tensor<1x256x2x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x256x2x8xf32>
    %185 = stablehlo.reshape %158 : (tensor<256x2x8xf32>) -> tensor<1x256x2x8xf32>
    %186 = stablehlo.dynamic_update_slice %83, %185, %11, %9, %9, %9 : (tensor<2x256x2x8xf32>, tensor<1x256x2x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x256x2x8xf32>
    %187 = stablehlo.reshape %170 : (tensor<256x4x8xf32>) -> tensor<256x2x2x8xf32>
    %188 = stablehlo.dot_general %187, %182, batching_dims = [1] x [1], contracting_dims = [3] x [2] : (tensor<256x2x2x8xf32>, tensor<256x2x8xf32>) -> tensor<2x256x2x256xf32>
    %189 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x256x2x256xf32>
    %190 = stablehlo.multiply %188, %189 : tensor<2x256x2x256xf32>
    %191 = stablehlo.broadcast_in_dim %24, dims = [1, 3] : (tensor<256x256xf32>) -> tensor<2x256x2x256xf32>
    %192 = stablehlo.add %190, %191 : tensor<2x256x2x256xf32>
    %193 = stablehlo.reduce(%192 init: %4) applies stablehlo.maximum across dimensions = [3] : (tensor<2x256x2x256xf32>, tensor<f32>) -> tensor<2x256x2xf32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [0, 1, 2] : (tensor<2x256x2xf32>) -> tensor<2x256x2x256xf32>
    %195 = stablehlo.subtract %192, %194 : tensor<2x256x2x256xf32>
    %196 = stablehlo.exponential %195 : tensor<2x256x2x256xf32>
    %197 = stablehlo.reduce(%196 init: %2) applies stablehlo.add across dimensions = [3] : (tensor<2x256x2x256xf32>, tensor<f32>) -> tensor<2x256x2xf32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [0, 1, 2] : (tensor<2x256x2xf32>) -> tensor<2x256x2x256xf32>
    %199 = stablehlo.divide %196, %198 : tensor<2x256x2x256xf32>
    %200 = stablehlo.dot_general %199, %158, batching_dims = [0] x [1], contracting_dims = [3] x [0] : (tensor<2x256x2x256xf32>, tensor<256x2x8xf32>) -> tensor<2x256x2x8xf32>
    %201 = stablehlo.transpose %200, dims = [1, 0, 2, 3] : (tensor<2x256x2x8xf32>) -> tensor<256x2x2x8xf32>
    %202 = stablehlo.reshape %201 : (tensor<256x2x2x8xf32>) -> tensor<256x32xf32>
    %203 = stablehlo.dot_general %202, %arg24, contracting_dims = [1] x [1] : (tensor<256x32xf32>, tensor<32x32xf32>) -> tensor<256x32xf32>
    %204 = stablehlo.add %129, %203 : tensor<256x32xf32>
    %205 = stablehlo.reduce(%204 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<256x32xf32>, tensor<f32>) -> tensor<256xf32>
    %206 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %207 = stablehlo.divide %205, %206 : tensor<256xf32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0] : (tensor<256xf32>) -> tensor<256x32xf32>
    %209 = stablehlo.subtract %204, %208 : tensor<256x32xf32>
    %210 = stablehlo.multiply %209, %209 : tensor<256x32xf32>
    %211 = stablehlo.reduce(%210 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<256x32xf32>, tensor<f32>) -> tensor<256xf32>
    %212 = stablehlo.divide %211, %206 : tensor<256xf32>
    %213 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %214 = stablehlo.add %212, %213 : tensor<256xf32>
    %215 = stablehlo.rsqrt %214 : tensor<256xf32>
    %216 = stablehlo.broadcast_in_dim %215, dims = [0] : (tensor<256xf32>) -> tensor<256x32xf32>
    %217 = stablehlo.multiply %209, %216 : tensor<256x32xf32>
    %218 = stablehlo.broadcast_in_dim %arg21, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %219 = stablehlo.multiply %217, %218 : tensor<256x32xf32>
    %220 = stablehlo.broadcast_in_dim %arg31, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %221 = stablehlo.add %219, %220 : tensor<256x32xf32>
    %222 = stablehlo.dot_general %221, %arg19, contracting_dims = [1] x [1] : (tensor<256x32xf32>, tensor<64x32xf32>) -> tensor<256x64xf32>
    %223 = stablehlo.dot_general %221, %arg22, contracting_dims = [1] x [1] : (tensor<256x32xf32>, tensor<64x32xf32>) -> tensor<256x64xf32>
    %224 = stablehlo.negate %222 : tensor<256x64xf32>
    %225 = stablehlo.exponential %224 : tensor<256x64xf32>
    %226 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<256x64xf32>
    %227 = stablehlo.add %226, %225 : tensor<256x64xf32>
    %228 = stablehlo.divide %226, %227 : tensor<256x64xf32>
    %229 = stablehlo.multiply %222, %228 : tensor<256x64xf32>
    %230 = stablehlo.multiply %229, %223 : tensor<256x64xf32>
    %231 = stablehlo.dot_general %230, %arg18, contracting_dims = [1] x [1] : (tensor<256x64xf32>, tensor<32x64xf32>) -> tensor<256x32xf32>
    %232 = stablehlo.add %204, %231 : tensor<256x32xf32>
    %233 = stablehlo.reduce(%232 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<256x32xf32>, tensor<f32>) -> tensor<256xf32>
    %234 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %235 = stablehlo.divide %233, %234 : tensor<256xf32>
    %236 = stablehlo.broadcast_in_dim %235, dims = [0] : (tensor<256xf32>) -> tensor<256x32xf32>
    %237 = stablehlo.subtract %232, %236 : tensor<256x32xf32>
    %238 = stablehlo.multiply %237, %237 : tensor<256x32xf32>
    %239 = stablehlo.reduce(%238 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<256x32xf32>, tensor<f32>) -> tensor<256xf32>
    %240 = stablehlo.divide %239, %234 : tensor<256xf32>
    %241 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %242 = stablehlo.add %240, %241 : tensor<256xf32>
    %243 = stablehlo.rsqrt %242 : tensor<256xf32>
    %244 = stablehlo.broadcast_in_dim %243, dims = [0] : (tensor<256xf32>) -> tensor<256x32xf32>
    %245 = stablehlo.multiply %237, %244 : tensor<256x32xf32>
    %246 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %247 = stablehlo.multiply %245, %246 : tensor<256x32xf32>
    %248 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %249 = stablehlo.add %247, %248 : tensor<256x32xf32>
    %250 = stablehlo.constant dense<1> : tensor<i32>
    %251 = stablehlo.subtract %arg34, %250 : tensor<i32>
    %252 = stablehlo.dynamic_slice %249, %251, %9, sizes = [1, 32] : (tensor<256x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x32xf32>
    %253 = stablehlo.reshape %252 : (tensor<1x32xf32>) -> tensor<32xf32>
    %254 = stablehlo.dot_general %253, %arg3, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<32x32xf32>) -> tensor<32xf32>
    return %254, %184, %186 : tensor<32xf32>, tensor<2x256x2x8xf32>, tensor<2x256x2x8xf32>
  }
}
