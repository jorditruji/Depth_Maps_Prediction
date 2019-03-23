from Data_management.dataset import Dataset

if __name__ == '__main__':
    # Testing:
    # Sample data
    from matplotlib import pyplot
    depths = ['../Test_samples/frame-000000.depth.pgm','../Test_samples/frame-000025.depth.pgm','../Test_samples/frame-000050.depth.pgm','../Test_samples/frame-000075.depth.pgm']
    dataset = Dataset(depths)

    print("Testing depth2RGB")
    rgb_frames = dataset.depth2RGB()
    print("Translated '{}' for '{}'".format(depths[0].split('/')[-1],rgb_frames[0].split('/')[-1]))

    # Test depth reader
    depth = read_depth(depths[-1])
    processed_depth, mask, real_depth = process_depth(depth,1)
    # Test jpg reader
    rgb = dataset.read_jpg_train(rgb_frames[-1])
    rgb_2 = dataset.read_jpg(rgb_frames[-1])
    print(rgb.size())


    # Matplotlib style display = channels last
    rgb= np.swapaxes(rgb.numpy(),0,-1)
    print(rgb.shape)
    rgb = np.swapaxes(rgb,0,1)
    print(rgb.shape)

    #Plot results
    f, axarr = pyplot.subplots(2, 2)
    axarr[0,0].imshow(rgb_2)#, 'gray', interpolation='nearest')
    axarr[0,0].set_title('Original RGB')
    axarr[0,1].imshow(rgb)
    axarr[0,1].set_title('Processed RGB')
    axarr[1,0].imshow(processed_depth,'gray', interpolation='nearest')
    axarr[1,0].set_title('Navier Stokes Impaint')


    # Test depth_gradient:
    gradient = dataset.imgrad(processed_depth)
    gradient = gradient.data.numpy()
    axarr[1,1].imshow(np.squeeze(gradient)/np.max(gradient),'gray', interpolation='nearest')
    axarr[1,1].set_title('Gradients Sobel')
    pyplot.show()
    print("Finished tests")
